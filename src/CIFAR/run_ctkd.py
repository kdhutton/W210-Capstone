import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import Models
from Dataset import CIFAR
from norm_dir_utils import colorstr, Save_Checkpoint, AverageMeter, DirectNormLoss, KDLoss
from utils.loss_functions import tkd_kdloss


import numpy as np
from pathlib import Path
import time
import json
import random
import logging
import argparse
import warnings
from torch.utils.tensorboard import SummaryWriter
import pdb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support


def compare_model_size(teacher, student):
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    return teacher_params, student_params

def compare_inference_time(teacher, student, dataloader):
    inputs, _ = next(iter(dataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    teacher = teacher.to(device)
    student = student.to(device)
    inputs = inputs.to(device)
    
    start_time = time.time()
    with torch.no_grad():
        teacher_outputs = teacher(inputs)
    teacher_time = time.time() - start_time

    start_time = time.time()
    with torch.no_grad():
        student_outputs = student(inputs)
    student_time = time.time() - start_time
    
    return teacher_time, student_time

def compare_performance_metrics(teacher, student, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher.eval()
    student.eval()
    
    all_labels = []
    all_teacher_preds = []
    all_student_preds = []

    for inputs, labels in dataloader:
        with torch.no_grad():
            teacher_outputs = teacher(inputs.to(device))
            student_outputs = student(inputs.to(device))
        all_labels.append(labels.cpu().numpy())
        all_teacher_preds.append(torch.argmax(teacher_outputs, dim=1).cpu().numpy())
        all_student_preds.append(torch.argmax(student_outputs, dim=1).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_teacher_preds = np.concatenate(all_teacher_preds)
    all_student_preds = np.concatenate(all_student_preds)
    
    metrics = {
        'accuracy': (accuracy_score(all_labels, all_teacher_preds), accuracy_score(all_labels, all_student_preds)),
        'precision': (precision_score(all_labels, all_teacher_preds, average='weighted', zero_division=0), precision_score(all_labels, all_student_preds, average='weighted', zero_division=0)),  # Updated line
        'recall': (recall_score(all_labels, all_teacher_preds, average='weighted'), recall_score(all_labels, all_student_preds, average='weighted')),
        'f1': (f1_score(all_labels, all_teacher_preds, average='weighted'), f1_score(all_labels, all_student_preds, average='weighted'))
    }

    return metrics

def get_temperature(epoch, initial_temperature, alpha):
    return initial_temperature / (1 + alpha * epoch)


def knowledge_distillation_loss(outputs, labels, teacher_outputs, temp, alpha):
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/temp, dim=1),
                                               F.softmax(teacher_outputs/temp, dim=1)) * (alpha * temp * temp) + \
           F.cross_entropy(outputs, labels) * (1. - alpha)


def train(student, teacher, train_dataloader, alpha, tkd_kdloss, optimizer, criterion, kd_criterion, args, epoch):
    train_loss = AverageMeter()
    train_error = AverageMeter()

    Ce_loss = AverageMeter()
    Kd_loss = AverageMeter()

    # Model on train mode
    student.train()
    teacher.eval()
    step_per_epoch = len(train_dataloader)

    for step, (images, labels) in enumerate(train_dataloader):
        start = time.time()
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()

        temperature = get_temperature(epoch, initial_temperature, alpha)

        # compute output
        student_outputs = student(images)
        with torch.no_grad():
            teacher_outputs = teacher(inputs)

        # Compute cross-entropy loss with true labels
        ce_loss = ce_criterion(student_outputs, labels)
        # Compute KL divergence loss with teacher outputs
        student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
        teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)
        kd_loss = kd_criterion(student_log_probs, teacher_probs)
        # Total loss
        loss = alpha * kd_loss + (1 - alpha) * ce_loss

        # measure accuracy and record loss
        batch_size = images.size(0)
        _, pred = student_outputs.data.cpu().topk(1, dim=1)
        train_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        train_loss.update(loss.item(), batch_size)

        Ce_loss.update(cls_loss.item(), batch_size)
        Kd_loss.update(div_loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        s1 = '\r{} [{}/{}]'.format(t, step+1, step_per_epoch)
        s2 = ' - {:.2f}ms/step - ce_loss: {:.3f} - kd_loss: {:.3f} -  - train_loss: {:.3f} - train_acc: {:.3f}'.format(
             1000 * (time.time() - start), ce_loss.item(), kd_loss.item(), train_loss.val, 1-train_error.val)

        print(s1+s2, end='', flush=True)

    print()
    return train_loss.avg, train_error.avg


def test(student, test_dataloader, criterion):
    test_loss = AverageMeter()
    test_error = AverageMeter()

    # Model on eval mode
    student.eval()

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.cuda(), labels.cuda()

            # compute logits
            logits = student(images, embed=False)

            loss = criterion(logits, labels)

            # measure accuracy and record loss
            batch_size = images.size(0)
            _, pred = logits.data.cpu().topk(1, dim=1)
            test_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
            test_loss.update(loss.item(), batch_size)
    

    return test_loss.avg, test_error.avg


def epoch_loop(student, teacher, train_set, test_set, args):
    # data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    # student
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # student = nn.DataParallel(student, device_ids=args.gpus)
    student = nn.DataParallel(student)
    student.to(device)
    # teacher = nn.DataParallel(teacher, device_ids=args.gpus)
    teacher = nn.DataParallel(teacher)
    teacher.to(device)

    # loss
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer
    optimizer = torch.optim.SGD(params=student.parameters(), lr=args.lr, 
                                momentum=args.momentum, weight_decay=args.weight_decay, 
                                nesterov=True)

    # weights
    save_dir = Path(args.save_dir)
    weights = save_dir / 'weights'
    weights.mkdir(parents=True, exist_ok=True)
    last = weights / 'last'
    best = weights / 'best'

    # acc,loss
    acc_loss = save_dir / 'acc_loss'
    acc_loss.mkdir(parents=True, exist_ok=True)

    train_acc_savepath = acc_loss / 'train_acc.npy'
    train_loss_savepath = acc_loss / 'train_loss.npy'
    val_acc_savepath = acc_loss / 'val_acc.npy'
    val_loss_savepath = acc_loss / 'val_loss.npy'

    # tensorboard
    logdir = save_dir / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    summary_writer = SummaryWriter(logdir, flush_secs=120)

    # resume
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        student.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_error = checkpoint['best_error']
        train_acc = checkpoint['train_acc']
        train_loss = checkpoint['train_loss']
        test_acc = checkpoint['test_acc']
        test_loss = checkpoint['test_loss']
        logger.info(colorstr('green', 'Resuming training from {} epoch'.format(start_epoch)))
    else:
        start_epoch = 0
        best_error = 0
        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []
        test_precision = []
        test_recall = []
        test_f1 = []

    # Train student
    best_error = 1
    ##
    patience = args.patience
    best_val_accuracy = 0
    best_val_loss = float('inf')
    epoch_val_losses = []
    epoch_val_accuracies = []
    ##
    for epoch in range(start_epoch, args.epochs):
        if epoch in [150, 180, 210]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        train_epoch_loss, train_error = train(student=student,
                                              teacher=teacher,
                                              train_dataloader=train_loader,
                                              optimizer=optimizer,
                                              criterion=criterion,
                                              kd_criterion = criterion,
                                              args=args,
                                              epoch=epoch)
        
        test_epoch_loss, test_error = test(student=student,
                                        test_dataloader=test_loader,
                                        criterion=criterion)

        s = "Train Loss: {:.3f}, Train Acc: {:.3f}, Test Loss: {:.3f}, Test Acc: {:.3f}, lr: {:.5f}".format(
            train_epoch_loss, 1-train_error, test_epoch_loss, 1-test_error, optimizer.param_groups[0]['lr'])
        logger.info(colorstr('green', s))

        # save acc,loss
        train_loss.append(train_epoch_loss)
        train_acc.append(1-train_error)
        test_loss.append(test_epoch_loss)
        test_acc.append(1-test_error)

        epoch_val_accuracies.append(1-test_error)
        epoch_val_losses.append(test_epoch_loss)

        # save student
        is_best = test_error < best_error
        best_error = min(best_error, test_error)
        state = {
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_error': best_error,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'test_loss': test_loss,
            }

        last_path = last / 'epoch_{}_loss_{:.3f}_acc_{:.3f}'.format(
            epoch + 1, test_epoch_loss, 1-test_error)
        best_path = best / 'epoch_{}_acc_{:.3f}'.format(
                epoch + 1, 1-best_error)

        Save_Checkpoint(state, last, last_path, best, best_path, is_best)

        # tensorboard
        if epoch == 1:
            images, labels = next(iter(train_loader))
            img_grid = torchvision.utils.make_grid(images)
            summary_writer.add_image('Cifar Image', img_grid)
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        summary_writer.add_scalar('train_loss', train_epoch_loss, epoch)
        summary_writer.add_scalar('train_error', train_error, epoch)
        summary_writer.add_scalar('val_loss', test_epoch_loss, epoch)
        summary_writer.add_scalar('val_error', test_error, epoch)

        ### 
        # Check if current validation combined loss is lower than the best combined loss
        if test_epoch_loss < best_val_loss:
            best_val_loss = test_epoch_loss
            best_val_accuracy = 1-test_error
            patience_counter = 0
        else:
            patience_counter += 1

        # decrease temperature for ctkd to soften output
        initial_temperature *= alpha

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            # print results
            best_checkpoint = torch.load(os.path.join(best_path, 'ckpt.pth'))['model_state_dict']
            student.load_state_dict(best_checkpoint)
            metrics = compare_performance_metrics(teacher, student, test_loader)
            teacher_time, student_time = compare_inference_time(teacher, student, test_loader)
            teacher_size, student_size = compare_model_size(teacher, student)

            final_report_banner = '- - - - - METRICS REPORT - - - - -'
            teacher_metrics = "TEACHER: accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}, teacher_inf: {:.3f}, teacher_size: {:.3f}".format(
                metrics['accuracy'][0], metrics['precision'][0], metrics['recall'][0], metrics['f1'][0], 
                teacher_time, teacher_size,)
            student_metrics = "STUDENT: accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}, student_inf: {:.3f}, student_size: {:.3f}".format(
                metrics['accuracy'][1], metrics['precision'][1], metrics['recall'][1], metrics['f1'][1], 
                 student_time, student_size)
            logger.info(colorstr('green', final_report_banner))
            logger.info(colorstr('green', teacher_metrics))
            logger.info(colorstr('green', student_metrics))
            break

        if epoch == (args.epochs - 1):
            best_checkpoint = torch.load(os.path.join(best_path, 'ckpt.pth'))['model_state_dict']
            student.load_state_dict(best_checkpoint)
            metrics = compare_performance_metrics(teacher, student, test_loader)
            teacher_time, student_time = compare_inference_time(teacher, student, test_loader)
            teacher_size, student_size = compare_model_size(teacher, student)

            final_report_banner = '- - - - - METRICS REPORT - - - - -'
            teacher_metrics = "TEACHER: accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}, teacher_inf: {:.3f}, teacher_size: {:.3f}".format(
                100*metrics['accuracy'][0], 100*metrics['precision'][0], 100*metrics['recall'][0], 100*metrics['f1'][0], 
                teacher_time, teacher_size,)
            student_metrics = "STUDENT: accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}, student_inf: {:.3f}, student_size: {:.3f}".format(
                100*metrics['accuracy'][1], 100*metrics['precision'][1], 100*metrics['recall'][1], 100*metrics['f1'][1], 
                 student_time, student_size)
            logger.info(colorstr('green', final_report_banner))
            logger.info(colorstr('green', teacher_metrics))
            logger.info(colorstr('green', student_metrics))
            
        
        ###

    summary_writer.close()
    if not os.path.exists(train_acc_savepath) or not os.path.exists(train_loss_savepath):
        np.save(train_acc_savepath, train_acc)
        np.save(train_loss_savepath, train_loss)
        np.save(val_acc_savepath, test_acc)
        np.save(val_loss_savepath, test_loss)

    

if __name__ == "__main__":
    student_names = sorted(name for name in Models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(Models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
    parser.add_argument('-f') # add to make this run in collab
    parser.add_argument("--student_name", type=str, default="resnet18_cifar", choices=student_names, help="student architecture")
    parser.add_argument("--dataset", type=str, default='cifar100')
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=128, help="batch size per gpu")
    parser.add_argument('--workers', default=32, type=int, help='number of data loading workers')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument("--teacher", type=str, default="resnet34_cifar", help="teacher architecture")
    parser.add_argument("--alpha", type=float, default=0.9, help="alpha for kd loss")
    parser.add_argument("--initial_temperature", type=float, default=20.0, help="temperature")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5, help='loss weight warm up epochs')

    # parser.add_argument("--gpus", type=list, default=[0, 1])
    
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument("--resume", type=str, help="best ckpt's path to resume most recent training")
    parser.add_argument("--save_dir", type=str, default="./run/CKD", help="save path, eg, acc_loss, weights, tensorboard, and so on")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [line:%(lineno)d] %(message)s', 
                        datefmt='%d %b %Y %H:%M:%S')
    logger = logging.getLogger(__name__)

    # args.batch_size = args.batch_size * len(args.gpus)
    args.batch_size = args.batch_size * 1

    # logger.info(colorstr('green', "Distribute train, gpus:{}, total batch size:{}, epoch:{}".format(args.gpus, args.batch_size, args.epochs)))
    logger.info(colorstr('green', "Distribute train, total batch size:{}, epoch:{}".format(args.batch_size, args.epochs)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 100

    train_set, test_set, num_class = CIFAR(name=args.dataset)
    student = models.resnet18(pretrained=False).to(device)
    student.fc = nn.Linear(512, num_classes)

    teacher =torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device)
    teacher.fc = nn.Linear(512, num_classes)
    


    logger.info(colorstr('green', 'Use ' + args.teacher + ' Training ' + args.student_name + ' ...'))
    # Train the student
    epoch_loop(student=student, teacher=teacher, train_set=train_set, test_set=test_set, args=args)

    