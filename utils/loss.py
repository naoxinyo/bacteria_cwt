from torch import nn
import torch.nn.functional as F

class DistillationLoss:                     #提高学生模型的性能
    def __init__(self, alpha): 
        self.student_loss = nn.CrossEntropyLoss() 
        self.distillation_loss = nn.KLDivLoss() 
        self.temperature = 1
        self.alpha = alpha

    def __call__(self, student_logits, student_target_loss, teacher_logits): 
        student_logits = F.softmax(student_logits, dim=1)
        teacher_logits = F.softmax(teacher_logits, dim=1)
        distillation_loss = self.distillation_loss(F.log_softmax(student_logits / self.temperature, dim=1), 
                                                   F.softmax(teacher_logits / self.temperature, dim=1)) 

        loss = (1 - self.alpha) * student_target_loss + self.alpha * distillation_loss 
        return loss