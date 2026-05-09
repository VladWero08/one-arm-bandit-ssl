import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from src.cnn13 import CNN13
from src.utils import update_ema_teacher, cons_kl, kl_to_uniform, mixup


def train_mean_teacher(
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    student: CNN13,
    teacher: CNN13,
    optimizer,
    device: torch.device,
    lambda_cons: float = 1.0,
    ema_decay: float = 0.999,
):
    student.train()
    teacher.eval()

    total_loss = 0
    total_sup = 0
    total_cons = 0

    for (xl, yl), (xu, _) in tqdm(zip(labeled_loader, unlabeled_loader)):
        xl, yl = xl.to(device), yl.to(device)
        xu = xu.to(device)

        optimizer.zero_grad()
        
        # compute supervised loss
        student_logits_l = student(xl)
        sup_loss = F.cross_entropy(student_logits_l, yl)

        # compute consistency loss
        student_logits_u = student(xu)
        with torch.no_grad():
            teacher_logits_u = teacher(xu)
        cons_loss = cons_kl(student_logits_u, teacher_logits_u)

        # aggregate the supervised and consistency loss
        loss = sup_loss + lambda_cons * cons_loss
        loss.backward()
        optimizer.step()

        update_ema_teacher(student, teacher, ema_decay)

        total_loss += loss.item()
        total_sup += sup_loss.item()
        total_cons += cons_loss.item()

    return {
        "total_loss": total_loss / len(unlabeled_loader),
        "sup_loss": total_sup / len(unlabeled_loader),
        "cons_loss": total_cons / len(unlabeled_loader),
    }


def train_rlgssl(
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    student: CNN13,
    teacher: CNN13,
    optimizer,
    device: torch.device,
    lambda_sup: float = 0.1,
    lambda_cons: float = 0.1,
    ema_decay: float = 0.999,
):
    student.train()
    teacher.eval()

    total_loss = 0
    total_rl_loss = 0
    total_sup_loss = 0
    total_cons_loss = 0

    for (xl, yl), (xu, _) in tqdm(zip(labeled_loader, unlabeled_loader)):
        xl, yl = xl.to(device), yl.to(device)
        xu = xu.to(device)
    
        # compute soft-labels for unlabeled data with the teacher
        with torch.no_grad():
            teacher_logits_u = teacher(xu)
            teacher_y_u = F.softmax(teacher_logits_u, dim=1)

        xm, ym = mixup(xl, yl, xu, teacher_y_u)    

        optimizer.zero_grad()
        
        # compute supervised loss 
        student_logits_l = student(xl)
        sup_loss = F.cross_entropy(student_logits_l, yl)

        # compute consistency loss 
        student_logits_u = student(xu)
        with torch.no_grad():
            teacher_logits_u = teacher(xu)
        cons_loss = cons_kl(student_logits_u, teacher_logits_u)

        # compute RL loss
        student_logits_m = student(xm)
        probs_m = F.softmax(student_logits_m, dim=1)

        reward = -F.mse_loss(probs_m, ym, reduction="none").mean(dim=1).detach()
        kl_coeff = kl_to_uniform(student_logits_u)
        rl_loss = -(kl_coeff * reward).mean() 

        # compute total loss
        loss = rl_loss + lambda_sup * sup_loss + lambda_cons * cons_loss
        loss.backward()
        optimizer.step()

        # update the teacher
        update_ema_teacher(student, teacher, ema_decay)

        # save the losses
        total_loss += loss.item()
        total_rl_loss += rl_loss.item()
        total_sup_loss += sup_loss.item()
        total_cons_loss += cons_loss.item()

    return {
        "total_loss": total_loss / len(unlabeled_loader),
        "rl_loss": total_rl_loss / len(unlabeled_loader),
        "sup_loss": total_sup_loss / len(unlabeled_loader),
        "cons_loss": total_cons_loss / len(unlabeled_loader),
    }


def train_rlgssl_plus(
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    student: CNN13,
    teacher: CNN13,
    optimizer,
    device: torch.device,
    lambda_sup: float = 0.1,
    lambda_cons: float = 0.1,
    lambda_entropy: float = 0.5,
    ema_decay: float = 0.999,
):
    student.train()
    teacher.eval()

    total_loss = 0
    total_rl_loss = 0
    total_sup_loss = 0
    total_cons_loss = 0

    for (xl, yl), (xu, _) in tqdm(zip(labeled_loader, unlabeled_loader)):
        xl, yl = xl.to(device), yl.to(device)
        xu = xu.to(device)
    
        # compute soft-labels for unlabeled data with the teacher
        with torch.no_grad():
            teacher_logits_u = teacher(xu)
            teacher_y_u = F.softmax(teacher_logits_u, dim=1)

        xm, ym = mixup(xl, yl, xu, teacher_y_u)  

        optimizer.zero_grad()
        
        # compute supervised loss 
        student_logits_l = student(xl)
        sup_loss = F.cross_entropy(student_logits_l, yl)

        # compute consistency loss 
        student_logits_u = student(xu)
        student_probs_u = F.softmax(student_logits_u, dim=1)
        with torch.no_grad():
            teacher_logits_u = teacher(xu)
        cons_loss = cons_kl(student_logits_u, teacher_logits_u)

        # compute RL loss
        student_logits_m = student(xm)
        probs_m = F.softmax(student_logits_m, dim=1)

        reward = -F.mse_loss(probs_m, ym, reduction="none").mean(dim=1).detach()
        entropy = -torch.sum(student_probs_u * torch.log(student_probs_u + 1e-8), dim=1)
        kl_coeff = kl_to_uniform(student_logits_u)
        rl_loss = -((kl_coeff + lambda_entropy * entropy) * reward).mean() 

        # compute total loss
        loss = rl_loss + lambda_sup * sup_loss + lambda_cons * cons_loss
        loss.backward()
        optimizer.step()

        # update the teacher
        update_ema_teacher(student, teacher, ema_decay)

        # save the losses
        total_loss += loss.item()
        total_rl_loss += rl_loss.item()
        total_sup_loss += sup_loss.item()
        total_cons_loss += cons_loss.item()

    return {
        "total_loss": total_loss / len(unlabeled_loader),
        "rl_loss": total_rl_loss / len(unlabeled_loader),
        "sup_loss": total_sup_loss / len(unlabeled_loader),
        "cons_loss": total_cons_loss / len(unlabeled_loader),
    }