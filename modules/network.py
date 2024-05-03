import torch
import torch.nn as nn
from torch.nn.functional import normalize


class SpatioTemporalFeatureExtractor(nn.Module):
    def __init__(self, batch_size, time_feat_dim, 
                    time_scaling_factor, mapped_feat_dim, 
                    road_graph_node_vec=None, 
                    device='cuda'):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.time_feat_dim = time_feat_dim
        self.time_scaling_factor = time_scaling_factor
        self.road_graph_node_vec = road_graph_node_vec

        if road_graph_node_vec is not None:
            num_inputs = road_graph_node_vec.shape[1] + self.time_feat_dim + 4
        else:
            num_inputs = 4
        num_hiddens = 256
        num_outputs = mapped_feat_dim

        self.fc = nn.Sequential(
                nn.Linear(num_inputs, num_hiddens), 
                nn.ReLU(),
                nn.Linear(num_hiddens, num_hiddens),
                nn.ReLU(),
                nn.Linear(num_hiddens, num_hiddens),
                nn.ReLU(),
                nn.Linear(num_hiddens, num_hiddens),
                nn.ReLU(),
                nn.Linear(num_hiddens, num_outputs),
                nn.ReLU(),
            )
        for i in range(0, 9, 2):
            nn.init.normal_(self.fc[i].weight.data, 0, 0.01)
            nn.init.constant_(self.fc[i].bias.data, 0)
        
    def forward(self, ts_info, road_graph_node_vec=None):
        tm, _, _, _ = torch.split(ts_info, 1, dim=1)
        coef = torch.arange(1, self.time_feat_dim + 1).to(self.device) * self.time_scaling_factor
        time_feat = tm * coef
        din = torch.cat((road_graph_node_vec, time_feat, ts_info), dim=-1)
        dout = self.fc(din)

        return dout
    
# for convenience to switch different mlp models
def get_st_proj(batch_size, time_feat_dim, time_scaling_factor, mapped_feat_dim, road_graph_node_vec, device, name):
    st_mlp_proj = SpatioTemporalFeatureExtractor(batch_size, time_feat_dim, time_scaling_factor, mapped_feat_dim, road_graph_node_vec, device) # default projector version

    st_proj_dict = {
        "MLP_emb": st_mlp_proj,
    }
    if name not in st_proj_dict.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
        
    return st_proj_dict[name]

class Network(nn.Module): # specific case of the MoCo architecture
    def __init__(self, st_proj):
        super(Network, self).__init__()
        self.st_proj = st_proj

    def forward(self, ts_info_i, ts_info_j, node_vec_i, node_vec_j):
        ts_feat_i = self.st_proj(ts_info_i, node_vec_i)
        ts_feat_j = self.st_proj(ts_info_j, node_vec_j)

        feat_i = normalize(ts_feat_i, dim=1)
        feat_j = normalize(ts_feat_j, dim=1)

        return feat_i, feat_j

    def extract_feat(self, ts, graph_emb):
        ts_feat = self.st_proj(ts, graph_emb)
        feat = normalize(ts_feat, dim=1)
    
        return feat

class Network_MoCo(nn.Module): # MoCo architecture: a query encoder, a key encoder, and a queue
    def __init__(self, encoder_q, encoder_k, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(Network_MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.mlp = mlp

        # create the encoders
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        if mlp:
            dim_mlp = self.encoder_q.fc[-2].weight.shape[1]
            self.mlp_projector_q = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_mlp))
            self.mlp_projector_k = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_mlp))

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        if self.mlp:
            for param_q, param_k in zip(self.mlp_projector_q.parameters(), self.mlp_projector_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x): # Batch shuffle, for making use of BatchNorm.
        # gather from all gpus
        batch_size = x.shape[0]

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        idx = idx_shuffle.view(1, -1)
    
        return x[idx], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle): # Undo batch shuffle.
        # restored index for this gpu
        idx = idx_unshuffle.view(1, -1)
        return x[idx]

    def forward(self, ts_info_i, ts_info_j, node_vec_i, node_vec_j):
        # compute query features
        q = self.encoder_q(ts_info_i, node_vec_i)  # a batch of query images: NxC
        if self.mlp:
            q = self.mlp_projector_q(q)  # nonlinear transformation
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(ts_info_j, node_vec_j)  # keys: NxC
            if self.mlp:
                k = self.mlp_projector_k(k)
            k = nn.functional.normalize(k, dim=1)

        # compute logits. Einstein sum is more intuitive
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1) # positive logits: Nx1
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()]) # negative logits: NxK

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, q, k
    
    def forward_extract_feat(self, ts_info, node_vec):
        # compute query features
        ts_feat = self.encoder_k(ts_info, node_vec)  # keys: NxC
        h = normalize(ts_feat, dim=1)

        return h
    
    def forward_extract_feat_gx(self, ts_info, node_vec):
        # compute query features
        ts_feat = self.encoder_k(ts_info, node_vec)  # keys: NxC
        if self.mlp:
            ts_feat = self.mlp_projector_k(ts_feat)
        h = normalize(ts_feat, dim=1)

        return h

