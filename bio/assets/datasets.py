"""Download pinned public dataset assets, retaining resumable partial files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import requests
from bio.io.paths import PROJECT_ROOT


def ranged_download(item, partial, connections=8):
    """Bound memory and preserve a contiguous resumable prefix while parallelizing."""
    offset = partial.stat().st_size if partial.exists() else 0
    total = item['size']
    if offset == total: return
    if offset > total: raise ValueError('Partial exceeds expected file size')
    chunk_size = 64*1024*1024
    folder = partial.with_suffix('.chunks')
    folder.mkdir(exist_ok=True)
    def fetch(start):
        end = min(total-1,start+chunk_size-1)
        path = folder/f'{start}-{end}'
        if path.exists() and path.stat().st_size == end-start+1: return path
        last_error = None
        for attempt in range(4):
            try:
                with requests.get(item['url'], params={'cellbench_range': str(start)}, headers={'Range':f'bytes={start}-{end}', 'Accept-Encoding':'identity'},
                                  stream=True,timeout=(45,120)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                        raise ValueError('Server did not honor requested byte range')
                    with path.open('wb') as handle:
                        for block in response.iter_content(4*1024*1024):handle.write(block)
                if path.stat().st_size != end-start+1:raise ValueError('Incomplete chunk')
                return path
            except Exception as exc:
                last_error=exc
                print(f"CHUNK RETRY {start} {attempt+1}: {exc}",flush=True)
                time.sleep(2)
        raise last_error
    starts=list(range(offset,total,chunk_size))
    ready={};failures=[];last=time.monotonic()
    with ThreadPoolExecutor(max_workers=connections) as executor:
        futures={executor.submit(fetch,start):start for start in starts}
        for future in as_completed(futures):
            try:ready[futures[future]]=future.result()
            except Exception as exc:failures.append(str(exc))
            if time.monotonic()-last>30:
                print(f"CHUNKS {item['id']} {len(ready)}/{len(starts)} ready",flush=True);last=time.monotonic()
    if failures:raise ValueError(f'{len(failures)} failed chunks: {failures[0]}')
    with partial.open('ab') as handle:
        for start in starts:
            path=ready[start]
            with path.open('rb') as chunk:
                for block in iter(lambda:chunk.read(8*1024*1024),b''):handle.write(block)
            path.unlink()
            if time.monotonic()-last>30:
                print(f"MERGE {item['id']} {handle.tell()/1e9:.2f} GB",flush=True);last=time.monotonic()
    folder.rmdir()


def download_asset(item):
    target = PROJECT_ROOT / item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_suffix('.download.json')
    if target.is_file() and receipt.is_file():
        record = json.loads(receipt.read_text())
        if target.stat().st_size == record['size'] and record['url'] == item['url']:
            return {'id': item['id'], 'status': 'existing', **record}
    partial = target.with_suffix(target.suffix+'.part')
    for attempt in range(4):
        try:
            if item.get('size',0)>8_000_000_000:
                ranged_download(item,partial)
            else:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {'Accept-Encoding':'identity'}
                if offset: headers['Range'] = f'bytes={offset}-'
                print(f"START {item['id']} offset={offset}", flush=True)
                with requests.get(item['url'], headers=headers, stream=True, timeout=(45,120)) as response:
                    response.raise_for_status()
                    if response.status_code == 206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected Content-Range')
                    append = offset > 0 and response.status_code == 206
                    received = 0
                    expected = response.headers.get('Content-Length')
                    last = time.monotonic()
                    with partial.open('ab' if append else 'wb') as handle:
                        for block in response.iter_content(8*1024*1024):
                            handle.write(block); received += len(block)
                            if time.monotonic()-last > 30:
                                print(f"PROGRESS {item['id']} {handle.tell()/1e9:.2f} GB",flush=True);last=time.monotonic()
                    if expected and received != int(expected): raise ValueError('Incomplete response')
            if item.get('size') and partial.stat().st_size != item['size']: raise ValueError('File size mismatch')
            hashes = {k:hashlib.new(k) for k in ('sha256','md5')}
            with partial.open('rb') as handle:
                magic = handle.read(8)
                if item.get('compression') == 'gzip':
                    if not magic.startswith(b'\x1f\x8b'): raise ValueError('Not a gzip file')
                elif magic != b'\x89HDF\r\n\x1a\n': raise ValueError('Not an HDF5 file')
                handle.seek(0)
                for block in iter(lambda: handle.read(8*1024*1024), b''):
                    for digest in hashes.values(): digest.update(block)
            if item.get('md5') and hashes['md5'].hexdigest() != item['md5']:
                partial.unlink(); raise ValueError('Official MD5 mismatch')
            if item.get('compression') != 'gzip':
                import h5py
                with h5py.File(partial,'r') as f:
                    if not all(k in f for k in ('X','obs','var')): raise ValueError('Missing AnnData components')
            partial.replace(target)
            record = dict(url=item['url'],size=target.stat().st_size,sha256=hashes['sha256'].hexdigest(),md5=hashes['md5'].hexdigest())
            receipt.write_text(json.dumps(record,indent=2)+'\n')
            print(f"DONE {item['id']} {record['size']/1e9:.2f} GB",flush=True)
            return dict(id=item['id'],status='downloaded',**record)
        except Exception as exc:
            print(f"RETRY {item['id']} {attempt+1}: {exc}",flush=True)
            if attempt==3:return dict(id=item['id'],status='failed',error=str(exc))
            time.sleep(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ids',nargs='+');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    assets=json.loads((PROJECT_ROOT/'resources/dataset_assets.json').read_text())
    if args.ids:
        missing=set(args.ids)-{a['id'] for a in assets}
        if missing:p.error(f'Unknown datasets: {sorted(missing)}')
        assets=[a for a in assets if a['id'] in args.ids]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results=list(executor.map(download_asset,assets))
    suffix = '_'.join(args.ids) if args.ids else 'all'
    path=PROJECT_ROOT/f'output/dataset_downloads_{suffix}.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results,indent=2)+'\n')
    return int(any(r['status']=='failed' for r in results))


if __name__=='__main__':raise SystemExit(main())
