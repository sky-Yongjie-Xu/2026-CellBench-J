"""Build reproducible all-gene RNA benchmark subsets from verified source files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData
from anndata._io.specs import read_elem
from bio.io.paths import PROJECT_ROOT


def stratified_rows(obs, keys, limit, seed):
    """Equal quota per observed stratum, then redistribute unused quota."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(v) for v in obs.reset_index(drop=True).groupby(keys, observed=True, sort=True).indices.values()]
    groups = [rng.permutation(v) for v in groups if len(v)]
    if len(groups) > limit: raise ValueError('More strata than requested cells')
    quota = np.zeros(len(groups), dtype=int)
    remaining = min(limit,len(obs))
    sizes = np.array([len(g) for g in groups])
    while remaining:
        active = np.flatnonzero(quota < sizes)
        step = max(1, remaining//len(active))
        for i in active:
            take = min(step,int(sizes[i]-quota[i]),remaining)
            quota[i] += take; remaining -= take
            if not remaining: break
    return np.sort(np.concatenate([g[:q] for g,q in zip(groups,quota)]))


def select_column(obs, candidates, required=True):
    for key in candidates:
        if key in obs: return key
    if required: raise ValueError(f'No column from {candidates}; found {list(obs.columns)}')
    return None


def matrix_slice(node, rows):
    if isinstance(node,h5py.Group):
        if node.attrs.get('encoding-type') == 'csc_matrix':
            nrows,ncols=map(int,node.attrs['shape'])
            lookup=np.full(nrows,-1,dtype=np.int32);lookup[rows]=np.arange(len(rows))
            indptr=node['indptr'][:]
            blocks=[]
            for start in range(0,ncols,256):
                end=min(start+256,ncols);lo,hi=int(indptr[start]),int(indptr[end])
                mapped=lookup[node['indices'][lo:hi]];keep=mapped>=0
                columns=np.searchsorted(indptr[start:end+1]-lo,np.flatnonzero(keep),side='right')-1
                counts=np.bincount(columns,minlength=end-start)
                ptr=np.r_[0,np.cumsum(counts)]
                values=node['data'][lo:hi][keep]
                blocks.append(sparse.csc_matrix((values,mapped[keep],ptr),shape=(len(rows),end-start)))
            return sparse.hstack(blocks,format='csr')
        indptr=node['indptr'][:]
        blocks=[];position=0
        while position<len(rows):
            first=int(rows[position]);endpos=max(position+1,int(np.searchsorted(rows,first+1024)))
            chosen=rows[position:endpos];last=int(chosen[-1])+1
            lo,hi=int(indptr[first]),int(indptr[last])
            block=sparse.csr_matrix((node['data'][lo:hi],node['indices'][lo:hi],indptr[first:last+1]-lo),shape=(last-first,int(node.attrs['shape'][1])))
            blocks.append(block[chosen-first]);position=endpos
        return sparse.vstack(blocks,format='csr')
    return sparse.csr_matrix(node[rows, :])


def valid_counts(values):
    for start in range(0,len(values),1_000_000):
        block=values[start:start+1_000_000]
        if not np.isfinite(block).all() or np.any(block<0) or not np.allclose(block,np.rint(block),atol=1e-5,rtol=0):return False
    return True


def prepare(item, recipe, limit, seed):
    source=PROJECT_ROOT/item['path']
    if not source.is_file() or not source.with_suffix('.download.json').is_file():
        raise FileNotFoundError(f'Unverified source: {source}')
    if item.get('compression')=='gzip':
        expanded=source.with_suffix('')
        if not expanded.is_file():
            part=expanded.with_suffix('.h5ad.part')
            with gzip.open(source,'rb') as src, part.open('wb') as dst: shutil.copyfileobj(src,dst,8*1024*1024)
            part.replace(expanded)
        source=expanded
    target=PROJECT_ROOT/'datasets'/recipe.get('filename',item['id']+'.h5ad')
    with h5py.File(source,'r') as f:
        if recipe.get('obs_columns'):
            group=f['obs'];index=read_elem(group[group.attrs.get('_index','_index')])
            columns={key:read_elem(group[key]) for key in recipe['obs_columns'] if key in group}
            obs=pd.DataFrame(columns,index=index)
        else:
            obs=read_elem(f['obs'])
        original_n=len(obs)
        label=select_column(obs,recipe['label'])
        donor=select_column(obs,recipe.get('donor',[]),required=False)
        batch=select_column(obs,recipe.get('batch',[]),required=False)
        obs['celltype']=obs[label].astype(str)
        if donor:obs['donor_id']=obs[donor].astype(str)
        if batch:obs['batch']=obs[batch].astype(str)
        perturb=recipe.get('perturbation',False)
        if perturb:
            condition=select_column(obs,recipe['condition'])
            values=obs[condition].astype(str)
            controls=values.isin(recipe['controls'])
            if not controls.any():raise ValueError('No annotated controls')
            if recipe.get('condition_split'):
                values=values.str.split(recipe['condition_split']).str[0]
            obs['condition']=values.where(~controls,'ctrl')
            obs['control']=controls.astype(int)
        valid=~obs['celltype'].str.strip().str.lower().isin(['nan','none','unknown','unassigned',''])
        if batch: valid &= ~obs['batch'].str.strip().str.lower().isin(['nan','none','unknown',''])
        original_positions=np.flatnonzero(valid.to_numpy())
        eligible=obs.iloc[original_positions].copy()
        keys=['celltype','condition'] if perturb else (['donor_id','celltype'] if donor else ['celltype'])
        if perturb and recipe.get('control_fraction'):
            control_pos=np.flatnonzero(eligible['control'].to_numpy()==1)
            treated_pos=np.flatnonzero(eligible['control'].to_numpy()==0)
            ncontrol=min(len(control_pos),max(1,int(limit*recipe['control_fraction'])))
            control_rows=control_pos[stratified_rows(eligible.iloc[control_pos],['celltype'],ncontrol,seed)]
            treated_rows=treated_pos[stratified_rows(eligible.iloc[treated_pos],keys,limit-len(control_rows),seed)]
            selected=np.sort(np.r_[control_rows,treated_rows])
        else:
            selected=stratified_rows(eligible,keys,limit,seed)
        rows=original_positions[selected]
        obs=obs.iloc[rows].copy()
        if not obs.index.is_unique:raise ValueError('Duplicate cell IDs')
        candidates=recipe.get('matrices',['layers/counts','raw/X','X'])
        x=None
        for path in candidates:
            if path not in f:continue
            node=f[path]
            is_csc=isinstance(node,h5py.Group) and node.attrs.get('encoding-type')=='csc_matrix'
            values=node['data'][:50000] if is_csc else matrix_slice(node,np.sort(rows[np.linspace(0,len(rows)-1,min(20,len(rows)),dtype=int)])).data
            if not valid_counts(values):continue
            # Chunk reads avoid materializing unrelated layers/embeddings.
            if is_csc:
                x=matrix_slice(node,rows).astype(np.float32)
            else:
                parts=[matrix_slice(node,rows[start:start+1000]) for start in range(0,len(rows),1000)]
                x=sparse.vstack(parts,format='csr',dtype=np.float32);del parts
            if not valid_counts(x.data):
                x=None;continue
            var=read_elem(f['raw/var'] if path=='raw/X' else f['var']);matrix=path;break
        if x is None:raise ValueError('No genuine nonnegative integer count matrix; do not round normalized data')
        feature_key=select_column(var,recipe.get('rna_feature_columns',[]),required=False)
        if feature_key:
            keep=var[feature_key].astype(str).isin(recipe['rna_feature_values']).to_numpy()
            if not keep.any():raise ValueError('No RNA features found')
            x=x[:,keep];var=var.iloc[np.flatnonzero(keep)].copy()
        gene_key=select_column(var,['gene_name','gene_symbols','feature_name','gene_symbol'],required=False)
        var['gene_name']=var[gene_key].astype(str) if gene_key else var.index.astype(str)
        ens=select_column(var,['gene_id','gene_ids','ensembl_id','ensemble_id','feature_id'],required=False)
        if ens:var['ensembl_id']=var[ens].astype(str)
        elif pd.Index(var.index.astype(str)).str.startswith('ENSG').all():var['ensembl_id']=var.index.astype(str)
        if not var.index.is_unique:raise ValueError('Duplicate feature IDs')
        if np.any(np.asarray(x.sum(axis=1)).ravel()<=0):raise ValueError('Zero-count cells')
        manifest=dict(id=item['id'],recipe=recipe,source=item['source'],url=item['url'],source_cells=original_n,eligible_cells=len(eligible),cells=len(obs),genes=len(var),matrix=matrix,seed=seed,limit=limit,strata=keys,selection='equal stratum quotas with redistribution; all genes retained',control_fraction=recipe.get('control_fraction'),label_column=label,donor_column=donor,batch_column=batch,cell_id_sha256=hashlib.sha256('\n'.join(obs.index.astype(str)).encode()).hexdigest())
        data=AnnData(x,obs=obs,var=var)
        data.uns['cellbench_provenance']=json.dumps(manifest)
        temporary=target.with_suffix('.prepared.h5ad')
        data.write_h5ad(temporary,compression='lzf')
    # Verify output before retiring any previous version.
    with h5py.File(temporary,'r') as f:
        if len(read_elem(f['obs']))!=manifest['cells']:raise ValueError('Output row count mismatch')
    if target.exists():
        old=dict(path=str(target.relative_to(PROJECT_ROOT)),size=target.stat().st_size,replaced_by=manifest)
        (PROJECT_ROOT/'datasets'/f'{target.stem}.replacement.json').write_text(json.dumps(old,indent=2)+'\n')
    temporary.replace(target)
    target.with_suffix('.provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ids',nargs='+',required=True)
    p.add_argument('--limit',type=int,default=30000);p.add_argument('--seed',type=int,default=41)
    args=p.parse_args()
    if args.limit<1:p.error('--limit must be positive')
    assets={a['id']:a for a in json.loads((PROJECT_ROOT/'resources/dataset_assets.json').read_text())}
    recipes=json.loads((PROJECT_ROOT/'resources/dataset_recipes.json').read_text())
    for name in args.ids:prepare(assets[name],recipes[name],args.limit,args.seed)


if __name__=='__main__':main()
