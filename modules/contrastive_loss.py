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

        self.negative_mask = self.mask_negative_samples(batch_size)
        self.positive_mask = self.mask_positive_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="mean").to(device)

    def mask_negative_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), device=self.device, dtype=torch.bool)
        mask.fill_diagonal_(False)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        return mask

    def mask_positive_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.zeros((N, N), device=self.device, dtype=torch.bool)
        for i in range(batch_size):
            mask[i, batch_size + i] = True
            mask[batch_size + i, i] = True
        return mask

    def forward(self, st_feat_i, st_feat_j, car_feat_i, car_feat_j, plate_feat_i, plate_feat_j, 
                gt_labels, cluster_result_centroids, cluster_result_concentration):
        # normalize features
        st_feat_i = F.normalize(st_feat_i, p=2, eps=1e-12, dim=-1)
        st_feat_j = F.normalize(st_feat_j, p=2, eps=1e-12, dim=-1)
        car_feat_i = F.normalize(car_feat_i, p=2, eps=1e-12, dim=-1)
        car_feat_j = F.normalize(car_feat_j, p=2, eps=1e-12, dim=-1)
        plate_feat_i = F.normalize(plate_feat_i, p=2, eps=1e-12, dim=-1)
        plate_feat_j = F.normalize(plate_feat_j, p=2, eps=1e-12, dim=-1)

        # concatenate features
        st_feat = torch.cat((st_feat_i, st_feat_j), dim=0)
        car_feat = torch.cat((car_feat_i, car_feat_j), dim=0)
        plate_feat = torch.cat((plate_feat_i, plate_feat_j), dim=0)

        # compute similarity
        st_feat_sim_mat = torch.matmul(st_feat, st_feat.T).contiguous()
        car_feat_sim_mat = torch.matmul(car_feat, car_feat.T).contiguous()
        plate_feat_sim_mat = torch.matmul(plate_feat, plate_feat.T).contiguous()
        
        # compute weighted similarities
        sim_mats = torch.stack([st_feat_sim_mat, car_feat_sim_mat, plate_feat_sim_mat], dim=0)
        weights = torch.tensor(self.weight, device=self.device).unsqueeze(1).unsqueeze(2).expand_as(sim_mats).clone()
        weights[sim_mats == 0] = 0
        weighted_sim_mat = torch.sum(sim_mats * weights, dim=0) / (torch.sum(weights, dim=0) + 1e-12)

        # apply temperature scaling and exponential function
        weighted_sim_mat = torch.exp(weighted_sim_mat / self.temperature)

        # extract positive scores using the positive mask
        positive_score = weighted_sim_mat[self.positive_mask].view(st_feat.size(0), -1)
        
        # negative similarity scores
        negative_sim = weighted_sim_mat[self.negative_mask].reshape(st_feat.size(0), -1)
        negative_sim_sum = negative_sim.sum(dim=-1)

        # basic contrastive loss
        loss = (-torch.log(positive_score / (positive_score + negative_sim_sum))).mean()

        # prototypical contrastive loss
        if not None in cluster_result_centroids and not None in cluster_result_concentration:
            # parse out the centroids and densities of different modalities
            cluster_result_st_feat_centroids, \
            cluster_result_car_feat_centroids, \
            cluster_result_plate_feat_centroids = cluster_result_centroids
            cluster_concentration = torch.stack(cluster_result_concentration)

            # organize negative prototypes indexes
            gt_labels_set = torch.unique(gt_labels)
            neg_proto_id_tensor = torch.stack([
                gt_labels_set[gt_labels_set != pos_proto_id] # filter out positive sample labels
                for pos_proto_id in gt_labels
                ])
            # combine positive and negative prototypes indexes
            proto_id_tensor = torch.cat([gt_labels, neg_proto_id_tensor], dim=1)

            # get selected prototypes
            st_feat_proto_selected = cluster_result_st_feat_centroids[proto_id_tensor]
            car_feat_proto_selected = cluster_result_car_feat_centroids[proto_id_tensor]
            plate_feat_proto_selected = cluster_result_plate_feat_centroids[proto_id_tensor]

            # compute multi-modal fusion prototypical logits with size sum(2NxDxN, dim=1) = 2NxN
            st_feat_logits_proto = torch.sum(st_feat.unsqueeze(2) * st_feat_proto_selected.transpose(1, 2), dim=1)
            car_feat_logits_proto = torch.sum(car_feat.unsqueeze(2) * car_feat_proto_selected.transpose(1, 2), dim=1)
            plate_feat_logits_proto = torch.sum(plate_feat.unsqueeze(2) * plate_feat_proto_selected.transpose(1, 2), dim=1)
            logits_proto = torch.stack([st_feat_logits_proto, car_feat_logits_proto, plate_feat_logits_proto], dim=0)
            
            # scale logits by applying the temperature parameters
            feat_temp_proto = cluster_concentration[:, proto_id_tensor]
            logits_proto /= (feat_temp_proto + 1e-12)
            
            # compute the fusion weight tensor
            weights = torch.tensor(self.weight, device=self.device).unsqueeze(1).unsqueeze(2).expand_as(logits_proto).clone()
            weights[(feat_temp_proto == 0) | (logits_proto == 0)] = 0

            # compute the logits
            logits_proto = torch.sum(logits_proto * weights, dim=0) / (torch.sum(weights, dim=0) + 1e-12)
            # targets for prototype assignment
            labels_proto = torch.zeros(logits_proto.size(0), device=self.device, dtype=torch.long)
            
            # calculate cross entropy loss for current batch
            loss_proto = self.criterion(logits_proto, labels_proto)
            # final ProtoNCE loss
            loss += loss_proto

        return loss
