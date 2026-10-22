from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
import torch
import scanpy as sc

data_name = "other"
split = "simulation"
batch_size = 64
eval_batch_size = 64

adata = sc.read_h5ad("/zangzelin/commusim/data_zoo/Used_data/adamson/perturb_processed.h5ad")
pert_data = PertData("/zangzelin/commusim/data_zoo/Used_data/adamson/")
pert_data.load(data_name=data_name,data_path="/zangzelin/commusim/data_zoo/Used_data/adamson/")
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)
train_loader = pert_data.dataloader["train_loader"]
for batch, batch_data in enumerate(train_loader):
    batch_size = len(batch_data.y)
    x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
    ori_gene_values = x[:, 0].view(batch_size, -1)
    pert_flags = x[:, 1].long().view(batch_size, -1)
    target_gene_values = batch_data.y  # (batch_size, n_genes)
    import pdb; pdb.set_trace()