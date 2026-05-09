import math
import torch
import torch.nn.functional as F

from src.cnn13 import CNN13


@torch.no_grad()
def update_ema_teacher(student: CNN13, teacher: CNN13, ema_decay: float) -> None:
    """
    Updates the parameters of the `teacher` model using using Exponential Moving Average
    and the `student`'s learned parameters.
    """
    student_params = dict(student.named_parameters())
    teacher_params = dict(teacher.named_parameters())
    for name in teacher_params:
        teacher_params[name].data.mul_(ema_decay).add_(student_params[name].data, alpha=1.0 - ema_decay)

    # also update the buffers because running statistics are used to normalize layers
    student_buffers = dict(student.named_buffers())
    teacher_buffers = dict(teacher.named_buffers())
    for name in teacher_buffers:
        teacher_buffers[name].data.copy_(student_buffers[name])


def cons_kl(student_logits, teacher_logits, temperature: float = 1.0):
    """
    Computs the KL-divergence for the given logits for student and teacher models.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def kl_to_uniform(logits):
    probs = F.softmax(logits, dim=1)
    c = probs.size(1)
    uniform = torch.full_like(probs, 1.0 / c)
    return F.kl_div(probs.clamp_min(1e-8).log(), uniform, reduction="none").sum(dim=1)


def mixup(
        xl: torch.Tensor, 
        yl: torch.Tensor, 
        xu: torch.Tensor, 
        yu_soft: torch.Tensor,
        num_classes: int = 100,    
    ) -> tuple[torch.tensor]:
    """
    Parameters
    ----------
    xl, yl: torch.Tensor, torch.Tensor
        This is the batch of labeled data
    xu, yu_soft: torch.Tensor, torch.Tensor
        This is the batch of unlabeled data and soft labels by the teacher
    """
    Nl, Nu = xl.shape[0], xu.shape[0]

    # oversample the images and labels
    ratio = math.ceil(Nu / Nl)
    # (batch, H, W, 3) ---> (batch * ratio, H, W, 3)
    xl_tiled = xl.repeat((ratio, 1, 1, 1))[:Nu]  
    yl_tiled = yl.repeat(ratio)[:Nu]
    
    # permutate the oversampled data
    perm = torch.randperm(Nu)
    xl_tiled = xl_tiled[perm]
    yl_tiled = yl_tiled[perm]
    yl_tiled_oh = F.one_hot(yl_tiled.long(), num_classes=num_classes).float()

    # generate the mixing parameters for each labeled image
    mu = torch.distributions.Beta(1.0, 1.0).sample((Nu, )).to(xu.device)
    mu_x = mu.view(Nu, 1, 1, 1)
    mu_y = mu.view(Nu, 1)

    xm = mu_x * xu + (1 - mu_x) * xl_tiled
    ym = mu_y * yu_soft + (1 - mu_y) * yl_tiled_oh

    return xm, ym