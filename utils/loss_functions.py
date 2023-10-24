import torch
import torch.nn as nn
import torch.nn.functional as F

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
