import torch
import torch.nn as nn
import torch.nn.functional as F

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
mse = nn.MSELoss()
smooth_l1 = nn.SmoothL1Loss()

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def Loss_interactive(outputs, teacher_outputs, T=2, interactive_type=0):
    if interactive_type==0:
        loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1))
    elif interactive_type==1:
        # Cosine distance
        loss = -torch.mean(cos(outputs, teacher_outputs))
    elif interactive_type==2:
        loss = mse(outputs, teacher_outputs)
    elif interactive_type == 3:
        loss = smooth_l1(outputs, teacher_outputs)
    else:
        raise Exception("Wrong interactive type!")
    return loss * (T * T) 
