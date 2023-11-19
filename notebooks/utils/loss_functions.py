import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def tkd_kdloss(student_outputs, teacher_outputs, temperature=1):
    """
    Compute the knowledge distillation loss.
    
    Args:
        student_outputs (torch.Tensor): Logits from the student network.
        teacher_outputs (torch.Tensor): Logits from the teacher network.
        temperature (float): Temperature parameter for knowledge distillation.
        
    Returns:
        torch.Tensor: Knowledge distillation loss.
    """
    student_softmax = F.log_softmax(student_outputs / temperature, dim=1)
    teacher_softmax = F.softmax(teacher_outputs / temperature, dim=1)
    
    kd_loss = nn.KLDivLoss(reduction='batchmean')(student_softmax, teacher_softmax)
    
    return kd_loss


def pairwise_distances(x, y):
    """
    Compute pairwise distances between the vectors in x and y
    Args:
    - x: A tensor of shape (batch_size, feature_dim)
    - y: A tensor of shape (batch_size, feature_dim)
    Returns:
    - pairwise_distances: Tensor of shape (batch_size, batch_size)
    """
    inner_product = torch.mm(x, y.t())
    x_norm = torch.norm(x, dim=1, keepdim=True)
    y_norm = torch.norm(y, dim=1, keepdim=True)
    distances = x_norm**2 - 2.0 * inner_product + y_norm.t()**2
    return distances

def DD_loss(student, teacher):
    """
    Distance-wise Distillation Loss
    """
    student_distances = pairwise_distances(student, student)
    teacher_distances = pairwise_distances(teacher, teacher)

    # We scale down the teacher distances to make it in range with the student's
    teacher_distances = teacher_distances / teacher_distances.detach().data.mean()
    student_distances = student_distances / student_distances.detach().data.mean()

    loss = F.mse_loss(student_distances, teacher_distances)
    return loss

def pairwise_angles(x, y):
    """
    Compute pairwise angles between the vectors in x and y
    """
    norm_x = torch.norm(x, dim=1, keepdim=True)
    norm_y = torch.norm(y, dim=1, keepdim=True)
    normalized_x = x / norm_x
    normalized_y = y / norm_y
    cosine_similarity = torch.mm(normalized_x, normalized_y.t())
    return cosine_similarity

def AD_loss(student, teacher):
    """
    Angle-wise Distillation Loss
    """
    student_angles = pairwise_angles(student, student)
    teacher_angles = pairwise_angles(teacher, teacher)

    loss = F.mse_loss(student_angles, teacher_angles)
    return loss

class RKDDistanceLoss(nn.Module):
    def __init__(self):
        super(RKDDistanceLoss, self).__init__()

    def forward(self, student, teacher):
        return F.pairwise_distance(student, teacher).mean()

class RKDAngleLoss(nn.Module):
    def __init__(self):
        super(RKDAngleLoss, self).__init__()

    def forward(self, student, teacher):
        # Normalize vectors
        student = F.normalize(student, dim=1)
        teacher = F.normalize(teacher, dim=1)

        # Get dot product between vectors
        dot = torch.matmul(student, teacher.t())
        return torch.mean(torch.acos(dot) ** 2)

def RKD_loss(student_outputs, teacher_outputs, criterion, alpha=0.1):
    """
    Compute the combined RKD loss.

    Args:
        student_outputs (torch.Tensor): Logits from the student network.
        teacher_outputs (torch.Tensor): Logits from the teacher network.
        criterion: The cross-entropy loss criterion.
        alpha (float): Weight for the RKD loss terms.

    Returns:
        torch.Tensor: Combined RKD loss.
    """
    distance_loss = F.pairwise_distance(student_outputs, teacher_outputs).mean()
    
    student_norm = F.normalize(student_outputs, dim=1)
    teacher_norm = F.normalize(teacher_outputs, dim=1)
    angle_loss = torch.mean(torch.acos(torch.clamp(torch.sum(student_norm * teacher_norm, dim=1), -1.0, 1.0)) ** 2)
    
    ce_loss = criterion(student_outputs, target)  # Replace 'target' with your actual target tensor
    
    combined_loss = ce_loss + alpha * (distance_loss + angle_loss)
    
    return combined_loss


def CTKD_loss(outputs, labels, teacher_outputs, temp, alpha):
    """
    Compute the knowledge distillation loss.

    Args:
        outputs (torch.Tensor): Logits from the student network.
        labels (torch.Tensor): Ground truth labels.
        teacher_outputs (torch.Tensor): Logits from the teacher network.
        temp (float): Temperature parameter for knowledge distillation.
        alpha (float): Weighting factor for the loss components.

    Returns:
        torch.Tensor: Knowledge distillation loss.
    """
    kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / temp, dim=1),
                                                  F.softmax(teacher_outputs / temp, dim=1)) * (alpha * temp * temp)
    ce_loss = F.cross_entropy(outputs, labels) * (1.0 - alpha)
    
    total_loss = kd_loss + ce_loss
    
    return total_loss

###########################################

# def DirectNormLoss(s_emb, t_emb, T_EMB, labels, num_class=100, nd_loss_factor=1.0):
    
#     def project_center(s_emb, t_emb, T_EMB, labels):
#         assert s_emb.size() == t_emb.size()
#         assert s_emb.shape[0] == len(labels)
#         loss = 0.0
#         for s, t, i in zip(s_emb, t_emb, labels):
#             i = i.item()
#             center = torch.tensor(T_EMB[str(i)]).cuda()
#             e_c = center / center.norm(p=2)
#             max_norm = max(s.norm(p=2), t.norm(p=2))
#             loss += 1 - torch.dot(s, e_c) / max_norm
#         return loss

#     nd_loss = self.project_center(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels) * nd_loss_factor
        
#     return nd_loss / len(labels)
    

# def KDLoss(s_out, t_out, kl_loss_factor=1.0, T=4.0):
#     '''
# 	Distilling the Knowledge in a Neural Network
# 	https://arxiv.org/pdf/1503.02531.pdf
# 	'''
#     kd_loss = F.kl_div(F.log_softmax(s_out / self.T, dim=1), 
#                            F.softmax(t_out / self.T, dim=1), 
#                            reduction='batchmean',
#                            ) * T * T
#     return kd_loss * kl_loss_factor



# def DKDLoss(s_logits, t_logits, labels, alpha=1.0, beta=1.0, T=4.0):
#     """Decoupled Knowledge Distillation(CVPR 2022)"""
    
#     def dkd_loss(s_logits, t_logits, labels):
#         gt_mask = get_gt_mask(s_logits, labels)
#         other_mask = get_other_mask(s_logits, labels)
#         s_pred = F.softmax(s_logits / T, dim=1)
#         t_pred = F.softmax(t_logits / T, dim=1)
#         s_pred = cat_mask(s_pred, gt_mask, other_mask)
#         t_pred = cat_mask(t_pred, gt_mask, other_mask)
#         s_log_pred = torch.log(s_pred)
#         tckd_loss = (
#             F.kl_div(s_log_pred, t_pred, size_average=False)
#             * (T**2)
#             / labels.shape[0]
#         )
#         pred_teacher_part2 = F.softmax(
#             t_logits / T - 1000.0 * gt_mask, dim=1
#         )
#         log_pred_student_part2 = F.log_softmax(
#             s_logits / T - 1000.0 * gt_mask, dim=1
#         )
#         nckd_loss = (
#             F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
#             * (T**2)
#             / labels.shape[0]
#         )
#         return alpha * tckd_loss + beta * nckd_loss

        
#     def get_gt_mask(logits, labels):
#         labels = labels.reshape(-1)
#         mask = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1).bool()
#         return mask
    
#     def get_other_mask(logits, labels):
#         labels = labels.reshape(-1)
#         mask = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0).bool()
#         return mask
    
#     def cat_mask(t, mask1, mask2):
#         t1 = (t * mask1).sum(dim=1, keepdims=True)
#         t2 = (t * mask2).sum(1, keepdims=True)
#         rt = torch.cat([t1, t2], dim=1)
#         return rt


#     loss = dkd_loss(s_logits, t_logits, labels)

#     return loss


####
class DirectNormLoss(nn.Module):

    def __init__(self, num_class=100, nd_loss_factor=1.0):
        super(DirectNormLoss, self).__init__()
        self.num_class = num_class
        self.nd_loss_factor = nd_loss_factor
    
    def project_center(self, s_emb, t_emb, T_EMB, labels):
        assert s_emb.size() == t_emb.size()
        assert s_emb.shape[0] == len(labels)
        loss = 0.0
        for s, t, i in zip(s_emb, t_emb, labels):
            i = i.item()
            center = torch.tensor(T_EMB[str(i)]).cuda()
            e_c = center / center.norm(p=2)
            max_norm = max(s.norm(p=2), t.norm(p=2))
            loss += 1 - torch.dot(s, e_c) / max_norm
        return loss
     
    def forward(self, s_emb, t_emb, T_EMB, labels):
        nd_loss = self.project_center(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels) * self.nd_loss_factor
        
        return nd_loss / len(labels)


class KDLoss(nn.Module):
    '''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
    def __init__(self, kl_loss_factor=1.0, T=4.0):
        super(KDLoss, self).__init__()
        self.T = T
        self.kl_loss_factor = kl_loss_factor

    def forward(self, s_out, t_out):
        kd_loss = F.kl_div(F.log_softmax(s_out / self.T, dim=1), 
                           F.softmax(t_out / self.T, dim=1), 
                           reduction='batchmean',
                           ) * self.T * self.T
        return kd_loss * self.kl_loss_factor


class DKDLoss(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, alpha=1.0, beta=1.0, T=4.0):
        super(DKDLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.T = T
    
    def dkd_loss(self, s_logits, t_logits, labels):
        gt_mask = self.get_gt_mask(s_logits, labels)
        other_mask = self.get_other_mask(s_logits, labels)
        s_pred = F.softmax(s_logits / self.T, dim=1)
        t_pred = F.softmax(t_logits / self.T, dim=1)
        s_pred = self.cat_mask(s_pred, gt_mask, other_mask)
        t_pred = self.cat_mask(t_pred, gt_mask, other_mask)
        s_log_pred = torch.log(s_pred)
        tckd_loss = (
            F.kl_div(s_log_pred, t_pred, size_average=False)
            * (self.T**2)
            / labels.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            t_logits / self.T - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            s_logits / self.T - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (self.T**2)
            / labels.shape[0]
        )
        return self.alpha * tckd_loss + self.beta * nckd_loss
    
    def get_gt_mask(self, logits, labels):
        labels = labels.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1).bool()
        return mask
    
    def get_other_mask(self, logits, labels):
        labels = labels.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0).bool()
        return mask
    
    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt
    
    def forward(self, s_logits, t_logits, labels):
        loss = self.dkd_loss(s_logits, t_logits, labels)

        return loss

