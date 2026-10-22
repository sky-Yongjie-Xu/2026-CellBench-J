import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tordata
from bio.vendor.gears.utils import loss_fct, uncertainty_loss_fct, parse_any_pert, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params
from torch_geometric.nn import SGConv



class MLP(nn.Module):
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if batch_norm and i < len(sizes) - 2:
                layers.append(nn.BatchNorm1d(sizes[i+1]))
            if i < len(sizes) - 2 or last_layer_act.lower() == 'relu':
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class PerturbationDecoder(nn.Module):
    def __init__(self, num_genes, hidden_size, num_go_layers, num_gene_layers, 
                 G_coexpress, G_coexpress_weight, G_go, G_go_weight,
                 num_perts, device):
        super().__init__()
        
        self.num_genes = num_genes
        self.hidden_size = hidden_size
        
        # 位置编码（基因）
        self.emb_pos = nn.Embedding(num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = nn.ModuleList([
            SGConv(hidden_size, hidden_size, K=1) for _ in range(num_gene_layers)
        ])
        
        # GNN 层用于扰动基因（GO图）
        self.sim_layers = nn.ModuleList([
            SGConv(hidden_size, hidden_size, K=1) for _ in range(num_go_layers)
        ])
        
        # 图数据
        self.G_coexpress = G_coexpress.to(device)
        self.G_coexpress_weight = G_coexpress_weight.to(device)
        self.G_go = G_go.to(device)
        self.G_go_weight = G_go_weight.to(device)
        
        # 扰动嵌入
        self.pert_emb = nn.Embedding(num_perts, hidden_size, max_norm=True)
        self.pert_fuse = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 基因嵌入批归一化
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        
        # 共享解码 MLP
        self.recovery_w = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size)
        )
        
        # 基因特异性参数（解码）
        self.indv_w1 = nn.Parameter(torch.randn(num_genes, hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.randn(num_genes, 1))
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # 跨基因 MLP
        self.cross_gene_state = nn.Sequential(
            nn.Linear(num_genes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.indv_w2 = nn.Parameter(torch.randn(1, num_genes, hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.randn(1, num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
    def forward(self, gene_emb, pert_idx, batch_size):
        """
        gene_emb: [batch_size*num_genes, hidden_size], 基因嵌入输入
        pert_idx: list of perturbed gene indices for each sample in batch
        batch_size: int
        """
        device = gene_emb.device
            
        # 基因位置编码
        pos_idx = torch.arange(self.num_genes, device=device).repeat(batch_size)
        pos_emb = self.emb_pos(pos_idx)
        
        # GNN编码基因位置 embedding（共表达图）
        for i, layer in enumerate(self.layers_emb_pos):
            pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
            if i < len(self.layers_emb_pos) - 1:
                pos_emb = F.relu(pos_emb)
        
        # 基因embedding融合位置编码
        base_emb = self.bn_emb(gene_emb) + 0.2 * pos_emb
        base_emb = F.relu(base_emb)
        
        # 计算扰动全局嵌入
        pert_emb_global = self.pert_emb.weight  # [num_perts, hidden_size]
        for i, layer in enumerate(self.sim_layers):
            pert_emb_global = layer(pert_emb_global, self.G_go, self.G_go_weight)
            if i < len(self.sim_layers) - 1:
                pert_emb_global = F.relu(pert_emb_global)
        
        # 聚合每个样本的扰动嵌入
        pert_track = {}
        for sample_idx, pert_list in enumerate(pert_idx):
            for p in pert_list:
                if p == -1:
                    continue
                idx = sample_idx * self.num_genes + p
                if idx not in pert_track:
                    pert_track[idx] = []
                pert_track[idx].append(pert_emb_global[p])
        
        # 融合扰动嵌入
        # for idx, emb_list in pert_track.items():
        #     if len(emb_list) == 1:
        #         fused_emb = self.pert_fuse(torch.cat([emb_list[0], emb_list[0]], dim=0).unsqueeze(0)).squeeze(0)
        #     else:
        #         fused_emb = self.pert_fuse(torch.stack(emb_list))
        #     base_emb[idx] += fused_emb
        for idx, emb_list in pert_track.items():
            if len(emb_list) == 1:
                fused_emb = self.pert_fuse(emb_list[0].unsqueeze(0)).squeeze(0)
            else:
                mean_emb = torch.mean(torch.stack(emb_list), dim=0)
                fused_emb = self.pert_fuse(mean_emb.unsqueeze(0)).squeeze(0)
            base_emb[idx] += fused_emb

        base_emb = self.bn_pert_base(base_emb)
        
        # 解码层
        out = self.recovery_w(base_emb)
        out = out.view(batch_size, self.num_genes, -1)
        out = out.unsqueeze(-1) * self.indv_w1  # 广播乘法
        w = out.sum(dim=2)
        out = w + self.indv_b1
        
        # 跨基因交互
        cross_gene_embed = self.cross_gene_state(out.squeeze(-1))
        cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes).view(batch_size, self.num_genes, -1)
        
        cross_gene_out = torch.cat([out, cross_gene_embed], dim=2)
        cross_gene_out = cross_gene_out * self.indv_w2
        cross_gene_out = cross_gene_out.sum(dim=2)
        out = cross_gene_out + self.indv_b2
        
        # 返回预测表达变化 [batch_size*num_genes]
        return out.view(batch_size * self.num_genes, 1).squeeze(-1)
