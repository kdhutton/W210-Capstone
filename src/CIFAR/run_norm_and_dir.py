import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import Models
from Models.embtrans_cifar import EmbTrans
from Dataset import CIFAR
from norm_dir_utils import colorstr, Save_Checkpoint, AverageMeter, DirectNormLoss, KDLoss

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
        _, teacher_outputs = teacher(inputs)
    teacher_time = time.time() - start_time

    start_time = time.time()
    with torch.no_grad():
        _, student_outputs = student(inputs)
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
            _, teacher_outputs = teacher(inputs.to(device))
            _, student_outputs = student(inputs.to(device))
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

def train(student, teacher, T_EMB, train_dataloader, optimizer, criterion, kd_loss, nd_loss, args, epoch):
    train_loss = AverageMeter()
    train_error = AverageMeter()

    Cls_loss = AverageMeter()
    Div_loss = AverageMeter()
    Norm_Dir_loss = AverageMeter()

    # Model on train mode
    student.train()
    teacher.eval()
    step_per_epoch = len(train_dataloader)

    for step, (images, labels) in enumerate(train_dataloader):
        start = time.time()
        images, labels = images.cuda(), labels.cuda()

        # compute output
        s_emb, s_logits = student(images, embed=True)

        with torch.no_grad():
            t_emb, t_logits = teacher(images, embed=True)

        # cls loss
        cls_loss = criterion(s_logits, labels) * args.cls_loss_factor
        # KD loss
        div_loss = kd_loss(s_logits, t_logits) * min(1.0, epoch/args.warm_up)
        # ND loss
        norm_dir_loss = nd_loss(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels)

        loss = cls_loss + div_loss + norm_dir_loss
        # measure accuracy and record loss
        batch_size = images.size(0)
        _, pred = s_logits.data.cpu().topk(1, dim=1)
        train_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        train_loss.update(loss.item(), batch_size)

        Cls_loss.update(cls_loss.item(), batch_size)
        Div_loss.update(div_loss.item(), batch_size)
        Norm_Dir_loss.update(norm_dir_loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        s1 = '\r{} [{}/{}]'.format(t, step+1, step_per_epoch)
        s2 = ' - {:.2f}ms/step - nd_loss: {:.3f} - kd_loss: {:.3f} - cls_loss: {:.3f} - train_loss: {:.3f} - train_acc: {:.3f}'.format(
             1000 * (time.time() - start), norm_dir_loss.item(), div_loss.item(), cls_loss.item(), train_loss.val, 1-train_error.val)

        print(s1+s2, end='', flush=True)

    print()
    return Norm_Dir_loss.avg, Div_loss.avg, Cls_loss.avg, train_loss.avg, train_error.avg


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
    kd_loss = KDLoss(kl_loss_factor=args.kd_loss_factor, T=args.t).to(device)
    nd_loss = DirectNormLoss(num_class=100, nd_loss_factor=args.nd_loss_factor).to(device)
    # optimizer
    optimizer = torch.optim.SGD(params=student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

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
        norm_dir_loss, div_loss, cls_loss, train_epoch_loss, train_error = train(student=student,
                                                                                 teacher=teacher,
                                                                                 T_EMB=T_EMB,
                                                                                 train_dataloader=train_loader,
                                                                                 optimizer=optimizer,
                                                                                 criterion=criterion,
                                                                                 kd_loss=kd_loss,
                                                                                 nd_loss=nd_loss,
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

        summary_writer.add_scalar('nd_loss', norm_dir_loss, epoch)
        summary_writer.add_scalar('kd_loss', div_loss, epoch)
        summary_writer.add_scalar('cls_loss', cls_loss, epoch)

        ### 
        # Check if current validation combined loss is lower than the best combined loss
        if test_epoch_loss < best_val_loss:
            best_val_loss = test_epoch_loss
            best_val_accuracy = 1-test_error
            patience_counter = 0
        else:
            patience_counter += 1

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
    parser.add_argument("--student_name", type=str, default="resnet20_cifar", choices=student_names, help="student architecture")
    parser.add_argument("--dataset", type=str, default='cifar100')
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=128, help="batch size per gpu")
    parser.add_argument('--workers', default=32, type=int, help='number of data loading workers')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--teacher", type=str, default="resnet56_cifar", help="teacher architecture")
    parser.add_argument("--teacher_weights", type=str, default="./ckpt/cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth", help="teacher weights path")
    parser.add_argument("--cls_loss_factor", type=float, default=1.0, help="cls loss weight factor")
    parser.add_argument("--kd_loss_factor", type=float, default=1.0, help="KD loss weight factor")
    parser.add_argument("--t", type=float, default=4.0, help="temperature")
    parser.add_argument("--nd_loss_factor", type=float, default=1.0, help="ND loss weight factor")
    parser.add_argument("--warm_up", type=float, default=20.0, help='loss weight warm up epochs')
    parser.add_argument("--patience", type=int, default=5, help='loss weight warm up epochs')


    # parser.add_argument("--gpus", type=list, default=[0, 1])
    
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument("--resume", type=str, help="best ckpt's path to resume most recent training")
    parser.add_argument("--save_dir", type=str, default="./run/KD++", help="save path, eg, acc_loss, weights, tensorboard, and so on")
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


    train_set, test_set, num_class = CIFAR(name=args.dataset)
    student = Models.__dict__[args.student_name](num_class=num_class)

    if args.student_name in ['wrn40_1_cifar', 'mobilenetv2', 'shufflev1_cifar', 'shufflev2_cifar']:
        student = EmbTrans(student=student, student_name=args.student_name)

    teacher = Models.__dict__[args.teacher](num_class=num_class)

    if args.teacher_weights:
        print('Load Teacher Weights')
        teacher_ckpt = torch.load(args.teacher_weights)['model']
        teacher.load_state_dict(teacher_ckpt)

        for param in teacher.parameters():
            param.requires_grad = False

    # res56    ./ckpt/teacher/resnet56/center_emb_train.json
    # res32x4  ./ckpt/teacher/resnet32x4/center_emb_train.json
    # wrn40_2  ./ckpt/teacher/wrn_40_2/center_emb_train.json
    # res50    ./ckpt/teacher/resnet50/center_emb_train.json
    # class-mean
    with open("./ckpt/teacher/resnet56/center_emb_train.json", 'r') as f:
        T_EMB = json.load(f)
    f.close()

    logger.info(colorstr('green', 'Use ' + args.teacher + ' Training ' + args.student_name + ' ...'))
    # Train the student
    epoch_loop(student=student, teacher=teacher, train_set=train_set, test_set=test_set, args=args)

    