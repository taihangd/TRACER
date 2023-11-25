import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalPrototypicalLoss(nn.Module):
    def __init__(self, batch_size, temperature, weight, device):
        super(MultiModalPrototypicalLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.weight = weight
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction="mean").to(device)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j, car_feat_i, car_feat_j, plate_feat_i, plate_feat_j, 
                gt_labels, cluster_result_centroids, cluster_result_densities):
        # normalize features
        z_i = F.normalize(z_i, p=2, eps=1e-12, dim=-1)
        z_j = F.normalize(z_j, p=2, eps=1e-12, dim=-1)
        car_feat_i = F.normalize(car_feat_i, p=2, eps=1e-12, dim=-1)
        car_feat_j = F.normalize(car_feat_j, p=2, eps=1e-12, dim=-1)
        plate_feat_i = F.normalize(plate_feat_i, p=2, eps=1e-12, dim=-1)
        plate_feat_j = F.normalize(plate_feat_j, p=2, eps=1e-12, dim=-1)

        # positive score
        st_feat_sim = torch.sum(z_i * z_j, dim=-1)
        car_feat_sim = torch.sum(car_feat_i * car_feat_j, dim=-1)
        plate_feat_sim = torch.sum(plate_feat_i * plate_feat_i, dim=-1)
        st_feat_weight = torch.ones_like(st_feat_sim, device=self.device) * self.weight[0]
        car_feat_weight = torch.ones_like(st_feat_sim, device=self.device) * self.weight[1]
        plate_feat_weight = torch.ones_like(st_feat_sim, device=self.device) * self.weight[2]
        st_feat_weight[st_feat_sim == 0] = 0
        car_feat_weight[car_feat_sim == 0] = 0
        plate_feat_weight[plate_feat_sim == 0] = 0
        positive_score = (st_feat_sim * st_feat_weight + car_feat_sim * car_feat_weight + \
                          plate_feat_sim * plate_feat_weight) / (st_feat_weight + car_feat_weight + plate_feat_weight + 1e-12)
        
        positive_score = torch.exp(positive_score / self.temperature)
        positive_score = torch.cat([positive_score, positive_score], dim=0)
        
        # negative scores
        z = torch.cat((z_i, z_j), dim=0)
        car_feat = torch.cat((car_feat_i, car_feat_j), dim=0)
        plate_feat = torch.cat((plate_feat_i, plate_feat_j), dim=0)
        st_feat_sim_mat = torch.matmul(z, z.T).contiguous()
        car_feat_sim_mat = torch.matmul(car_feat, car_feat.T).contiguous()
        plate_feat_sim_mat = torch.matmul(plate_feat, plate_feat.T).contiguous()
        st_feat_weight = torch.ones_like(st_feat_sim_mat, device=self.device) * self.weight[0]
        car_feat_weight = torch.ones_like(st_feat_sim_mat, device=self.device) * self.weight[1]
        plate_feat_weight = torch.ones_like(st_feat_sim_mat, device=self.device) * self.weight[2]
        st_feat_weight[st_feat_sim_mat == 0] = 0
        car_feat_weight[car_feat_sim_mat == 0] = 0
        plate_feat_weight[plate_feat_sim_mat == 0] = 0
        sim_mat = (st_feat_sim_mat * st_feat_weight + car_feat_sim_mat * car_feat_weight + \
                plate_feat_sim_mat * plate_feat_weight) / (st_feat_weight + car_feat_weight + plate_feat_weight + 1e-12)
        sim_mat = torch.exp(sim_mat / self.temperature)
        
        batch_size = min(self.batch_size, z_i.shape[0])
        feat_num = 2 * batch_size
        negative_sim = sim_mat[self.mask].reshape(feat_num, -1)
        negative_sim_sum = negative_sim.sum(dim = -1)
        
        # basic contrastive loss
        loss = (-torch.log(positive_score / (positive_score + negative_sim_sum))).mean()

        # prototypical contrastive loss
        if not None in cluster_result_centroids and not None in cluster_result_densities:
            # parse out the centroids and densities of different modalities
            cluster_result_st_feat_centroids, \
            cluster_result_car_feat_centroids, \
            cluster_result_plate_feat_centroids = cluster_result_centroids
            cluster_result_st_feat_density, \
            cluster_result_car_feat_density, \
            cluster_result_plate_feat_density = cluster_result_densities

            # organize negative prototypes indexes
            gt_labels_set = set(gt_labels.cpu().numpy().flatten())
            neg_proto_id_lists = [list(gt_labels_set - {pos_proto_id.item()}) for i, pos_proto_id in enumerate(gt_labels)]
            neg_proto_id_tensor = torch.tensor(neg_proto_id_lists, dtype=torch.int32, device=self.device)
            # combine positive and negative prototypes indexes
            proto_id_tensor = torch.cat([gt_labels, neg_proto_id_tensor], dim=1)
            # get selected prototypes
            st_feat_proto_selected = cluster_result_st_feat_centroids[proto_id_tensor]
            car_feat_proto_selected = cluster_result_car_feat_centroids[proto_id_tensor]
            plate_feat_proto_selected = cluster_result_plate_feat_centroids[proto_id_tensor]

            # compute multi-modal fusion prototypical logits (the similarity score to each prototype)
            st_feat_logits_proto = torch.sum(z.unsqueeze(2) * st_feat_proto_selected.transpose(1, 2), dim=1)
            car_feat_logits_proto = torch.sum(car_feat.unsqueeze(2) * car_feat_proto_selected.transpose(1, 2), dim=1)
            plate_feat_logits_proto = torch.sum(plate_feat.unsqueeze(2) * plate_feat_proto_selected.transpose(1, 2), dim=1)
            
            # targets for prototype assignment
            labels_proto = torch.zeros(feat_num, device=self.device).long()
            # scaling temperatures for the selected prototypes
            cluster_id_selected = torch.cat([gt_labels, neg_proto_id_tensor], dim=1)
            st_feat_temp_proto = cluster_result_st_feat_density[cluster_id_selected]
            car_feat_temp_proto = cluster_result_car_feat_density[cluster_id_selected]
            plate_feat_temp_proto = cluster_result_plate_feat_density[cluster_id_selected]
            st_feat_logits_proto /= (st_feat_temp_proto + 1e-12)
            car_feat_logits_proto /= (car_feat_temp_proto + 1e-12)
            plate_feat_logits_proto /= (plate_feat_temp_proto + 1e-12)

            # compute the fusion weight tensor
            st_feat_weight = torch.ones_like(st_feat_logits_proto, device=self.device) * self.weight[0]
            car_feat_weight = torch.ones_like(st_feat_logits_proto, device=self.device) * self.weight[1]
            plate_feat_weight = torch.ones_like(st_feat_logits_proto, device=self.device) * self.weight[2]
            st_feat_weight[st_feat_temp_proto == 0] = 0
            car_feat_weight[car_feat_temp_proto == 0] = 0
            plate_feat_weight[plate_feat_temp_proto == 0] = 0
            st_feat_weight[st_feat_logits_proto == 0] = 0
            car_feat_weight[car_feat_logits_proto == 0] = 0
            plate_feat_weight[plate_feat_logits_proto == 0] = 0
            logits_proto = (st_feat_logits_proto * st_feat_weight + car_feat_logits_proto * car_feat_weight + \
                            plate_feat_logits_proto * plate_feat_weight) / (st_feat_weight + car_feat_weight + plate_feat_weight + 1e-12)

            loss_proto = self.criterion(logits_proto, labels_proto)  
            loss += loss_proto

        return loss
