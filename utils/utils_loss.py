import torch
import torch.nn.functional as F
import torch.nn as nn


class ConLoss(nn.Module):
    def __init__(self, predicted_score, base_temperature=0.07):
        super().__init__()
        self.predicted_score = predicted_score
        self.init_predicted_score = predicted_score.detach()
        self.base_temperature = base_temperature
    
    
    def forward(self, args, outputs, features, Y, index):
        batch_size = args.batch_size
        
        device = torch.device('cuda')
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), args.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) 
        
        logits = anchor_dot_contrast - logits_max.detach() 
                
        Y = Y.float()
        
        output_sm = F.softmax(outputs[0 : batch_size, :], dim=1).float()
        output_sm_d = output_sm.detach()
        _, target_predict = (output_sm_d * Y).max(1)

        predict_labels =  target_predict.repeat(batch_size, 1).to(device)
        
        mask_logits = torch.zeros_like(predict_labels).float().to(device)
        
        pos_set = (Y == 1.0).nonzero().to(device)
        
        ones_flag = torch.ones(batch_size).float().to(device)
        zeros_flag = torch.zeros(batch_size).float().to(device)
        
        for pos_set_i in range(pos_set.shape[0]):
            sample_idx = pos_set[pos_set_i][0]
            class_idx = pos_set[pos_set_i][1]
            mask_logits_tmp = torch.where(predict_labels[sample_idx] == class_idx, ones_flag, zeros_flag).float()
            if mask_logits_tmp.sum() > 0:
                mask_logits_tmp = mask_logits_tmp / mask_logits_tmp.sum()
                mask_logits[sample_idx] = mask_logits[sample_idx] + mask_logits_tmp * self.predicted_score[sample_idx][class_idx]

        mask_logits = mask_logits.repeat(anchor_count, contrast_count)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask_logits),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        ).float()

        mask_logits = mask_logits * logits_mask
        exp_logits = logits_mask * torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask_logits * log_prob).sum(1)
        
        loss_con_m = - (args.temperature / self.base_temperature) * mean_log_prob_pos
        loss_con = loss_con_m.view(anchor_count, batch_size).mean()
        
        revisedY_raw = Y.clone()
        revisedY_raw = revisedY_raw * output_sm_d
        revisedY_raw = revisedY_raw / revisedY_raw.sum(dim = 1).repeat(args.num_class, 1).transpose(0, 1)
        new_target = revisedY_raw.detach()
        
        return loss_con, new_target
    
    
    def update_target(self, batch_index, updated_confidence):
        with torch.no_grad():
            self.predicted_score[batch_index, :] = updated_confidence.detach()
        return None



class ClsLoss(nn.Module):
    def __init__(self, predicted_score):
        super().__init__()
        self.predicted_score = predicted_score
        self.init_predicted_score = predicted_score.detach()

    
    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = self.predicted_score[index, :] * logsm_outputs 
        cls_loss = - ((final_outputs).sum(dim=1)).mean()
        return cls_loss
    
    
    def update_target(self, batch_index, updated_confidence):
        with torch.no_grad():
            self.predicted_score[batch_index, :] = updated_confidence.detach()
        return None

    