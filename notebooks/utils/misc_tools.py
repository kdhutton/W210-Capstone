import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import models_package
from pathlib import Path
from collections import OrderedDict
import os, shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
import models_package
from utils.loss_functions import tkd_kdloss, DD_loss, AD_loss, RKDDistanceLoss, RKDAngleLoss, DKDLoss, DirectNormLoss, KDLoss
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'





#### finding the optimal learning rate

##### HELPER FUNCTION FOR FEATURE EXTRACTION

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


def best_LR_nd(save_name, model, trainloader, criterion, optimizer, scheduler, 
                num_epochs=5, emb = False, lr_range=(1e-4, 1e-1), plot_loss=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    # create hook for feature embeddings
    model.avgpool.register_forward_hook(get_features('feats'))
    model.to(device)
    lr_values = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), num_epochs * len(trainloader))  # Generate learning rates for each batch
    lr_iter = iter(lr_values)
    losses = []
    lrs = []
    
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            lr = next(lr_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr  # Set new learning rate
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # the Norm and Direction models give 2 outputs - feature embeddings and output
            if emb:
                outputs = model(inputs)
                feats = features['feats']
                emb_feats = torch.flatten(feats, 1)

                
                # _, outputs = model(inputs)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            lrs.append(lr)
    
    # Calculate the derivative of the loss
    loss_derivative = np.gradient(losses)
    
    # Find the learning rate corresponding to the minimum derivative (steepest decline)
    best_lr_index = np.argmin(loss_derivative)
    best_lr = lrs[best_lr_index]

    
    plot_path = './figs/LR/'
    os.makedirs(plot_path, exist_ok=True)
    plot_name = str(plot_path + save_name)

    
    if plot_loss:
        plt.figure()
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Range Test')
        plt.axvline(x=best_lr, color='red', linestyle='--', label=f'Best LR: {best_lr}')
        plt.legend()
        plt.savefig(plot_name, bbox_inches='tight')
        plt.show()
    
    print(f'Best learning rate: {best_lr}')
    return best_lr


def best_LR(save_name, model, trainloader, criterion, optimizer, scheduler, 
                num_epochs=5, emb = False, lr_range=(1e-4, 1e-1), plot_loss=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    lr_values = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), num_epochs * len(trainloader))  # Generate learning rates for each batch
    lr_iter = iter(lr_values)
    losses = []
    lrs = []
    
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            lr = next(lr_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr  # Set new learning rate
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # the Norm and Direction models give 2 outputs - feature embeddings and output
            if emb:
                _, outputs = model(inputs)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            lrs.append(lr)
    
    # Calculate the derivative of the loss
    loss_derivative = np.gradient(losses)
    
    # Find the learning rate corresponding to the minimum derivative (steepest decline)
    best_lr_index = np.argmin(loss_derivative)
    best_lr = lrs[best_lr_index]

    
    plot_path = './figs/LR/'
    os.makedirs(plot_path, exist_ok=True)
    plot_name = str(plot_path + save_name)

    
    if plot_loss:
        plt.figure()
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Range Test')
        plt.axvline(x=best_lr, color='red', linestyle='--', label=f'Best LR: {best_lr}')
        plt.legend()
        plt.savefig(plot_name, bbox_inches='tight')
        plt.show()
    
    print(f'Best learning rate: {best_lr}')
    return best_lr



def plot_loss_curve(losses):
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()


def train_teacher(model_name, model, trainloader, criterion, optimizer, scheduler, num_epochs=240, patience=5):
    ''' A function to train the teacher models'''

    best_val_loss = float('inf')
    patience_counter = 0
    epoch_losses = [] 
    val_losses = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    best_train_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0  
        num_batches = 0  
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            if i % 100 == 99:  # Print every 100 mini-batches
                # print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        epoch_loss /= num_batches  
        epoch_losses.append(epoch_loss)

        
        model.eval()
        total_correct = 0
        total_samples = 0
        total_val_loss = 0.0
        num_batches = 0  
        with torch.no_grad():
            for inputs, labels in tqdm(testloader):
                val_inputs, val_labels = inputs.to(device), labels.to(device)
                # val_inputs = val_data['img'].to(device)
                # val_labels = val_data['label'].to(device)
    
                # Forward pass for validation
                _, val_outputs = model(val_inputs)
    
                val_loss = criterion(val_outputs, val_labels)

                total_val_loss += val_loss.item()
    
                # Compute the validation accuracy
                _, predicted = torch.max(val_outputs, 1)
                total_samples += val_labels.size(0)
                total_correct += (predicted == val_labels).sum().item()
                num_batches += 1
            
            total_val_loss /= num_batches
            val_losses.append(total_val_loss)
            accuracy = total_correct / total_samples
            print(f'*****Epoch {epoch + 1}/{num_epochs}*****\n' 
            f'*****Train Loss: {epoch_loss: .6f} Val Loss: {total_val_loss: .6f}*****\n'
            f'*****Validation Accuracy: {accuracy * 100:.2f}%*****\n')

        
        # Check for early stopping
        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            patience_counter = 0 
            
            # checkpoint
            save_path = './weights/'

            model_save_path = os.path.join(save_path, model_name)
            
            os.makedirs(model_save_path, exist_ok=True)
        
            model_save_name = os.path.join(model_save_path, 'checkpoint.pth')
            mode_weights_name = os.path.join(model_save_path, 'weights.pth')
        
            torch.save(model.state_dict(), mode_weights_name)
            torch.save(model, model_save_name)
            
            # model_save_name = str(save_path + model_name + '/checkpoint.pth')
            # mode_weights_name = str(save_path + model_name + '/weights.pth')

            # torch.save(model.state_dict(), mode_weights_name)
            # torch.save(model, model_save_name)

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping')
            break

        scheduler.step()

    print("Finished Training Teacher")
    plot_loss_curve(val_losses)
    return model

def train_teacher_efficientnet(model_name, model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=240, patience=5):
    ''' A function to train the teacher models'''

    best_val_loss = float('inf')
    patience_counter = 0
    epoch_losses = [] 
    val_losses = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    # model.avgpool.register_forward_hook(get_features('feats'))

    best_train_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0  
        num_batches = 0  
        features = {}
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # feats = features['feats'].cpu().numpy()
            # emb_feats = feats.flatten()
            # _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            if i % 100 == 99:  # Print every 100 mini-batches
                # print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        epoch_loss /= num_batches  
        epoch_losses.append(epoch_loss)

        
        model.eval()
        total_correct = 0
        total_samples = 0
        total_val_loss = 0.0
        num_batches = 0  
        with torch.no_grad():
            for inputs, labels in tqdm(testloader):
                val_inputs, val_labels = inputs.to(device), labels.to(device)
                # val_inputs = val_data['img'].to(device)
                # val_labels = val_data['label'].to(device)
    
                # Forward pass for validation
                # _, val_outputs = model(val_inputs)
                val_outputs = model(val_inputs)
    
                val_loss = criterion(val_outputs, val_labels)

                total_val_loss += val_loss.item()
    
                # Compute the validation accuracy
                _, predicted = torch.max(val_outputs, 1)
                total_samples += val_labels.size(0)
                total_correct += (predicted == val_labels).sum().item()
                num_batches += 1
            
            total_val_loss /= num_batches
            val_losses.append(total_val_loss)
            accuracy = total_correct / total_samples
            print(f'*****Epoch {epoch + 1}/{num_epochs}*****\n' 
            f'*****Train Loss: {epoch_loss: .6f} Val Loss: {total_val_loss: .6f}*****\n'
            f'*****Validation Accuracy: {accuracy * 100:.2f}%*****\n')

        
        # Check for early stopping
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            patience_counter = 0 
            
            # checkpoint
            save_path = './weights/'

            model_save_path = os.path.join(save_path, model_name)
            
            os.makedirs(model_save_path, exist_ok=True)
        
            model_save_name = os.path.join(model_save_path, 'checkpoint.pth')
            mode_weights_name = os.path.join(model_save_path, 'weights.pth')
        
            torch.save(model.state_dict(), mode_weights_name)
            torch.save(model, model_save_name)
            
            # model_save_name = str(save_path + model_name + '/checkpoint.pth')
            # mode_weights_name = str(save_path + model_name + '/weights.pth')

            # torch.save(model.state_dict(), mode_weights_name)
            # torch.save(model, model_save_name)

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping')
            break

        scheduler.step()

    print("Finished Training Teacher")
    plot_loss_curve(val_losses)
    return model

def train_teacher_efficientnet_wider(model_name, model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=240, patience=5):
    ''' A function to train the teacher models'''

    best_val_loss = float('inf')
    patience_counter = 0
    epoch_losses = [] 
    val_losses = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    # model.avgpool.register_forward_hook(get_features('feats'))

    best_train_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0  
        num_batches = 0  
        features = {}
        for index, data in enumerate(tqdm(trainloader)):
            inputs = data['img'].to(device)
            labels = data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # feats = features['feats'].cpu().numpy()
            # emb_feats = feats.flatten()
            # _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            if index % 100 == 99:  # Print every 100 mini-batches
                # print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        epoch_loss /= num_batches  
        epoch_losses.append(epoch_loss)

        
        model.eval()
        total_correct = 0
        total_samples = 0
        total_val_loss = 0.0
        num_batches = 0  
        with torch.no_grad():
            for index, data in enumerate(tqdm(testloader)):

                val_inputs = data['img'].to(device)
                val_labels = data['label'].to(device)
    
                # Forward pass for validation
                # _, val_outputs = model(val_inputs)
                val_outputs = model(val_inputs)
    
                val_loss = criterion(val_outputs, val_labels)

                total_val_loss += val_loss.item()
    
                # Compute the validation accuracy
                _, predicted = torch.max(val_outputs, 1)
                total_samples += val_labels.size(0)
                total_correct += (predicted == val_labels).sum().item()
                num_batches += 1
            
            total_val_loss /= num_batches
            val_losses.append(total_val_loss)
            accuracy = total_correct / total_samples
            print(f'*****Epoch {epoch + 1}/{num_epochs}*****\n' 
            f'*****Train Loss: {epoch_loss: .6f} Val Loss: {total_val_loss: .6f}*****\n'
            f'*****Validation Accuracy: {accuracy * 100:.2f}%*****\n')

        
        # Check for early stopping
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            patience_counter = 0 
            
            # checkpoint
            save_path = './weights/'

            model_save_path = os.path.join(save_path, model_name)
            
            os.makedirs(model_save_path, exist_ok=True)
        
            model_save_name = os.path.join(model_save_path, 'checkpoint.pth')
            mode_weights_name = os.path.join(model_save_path, 'weights.pth')
        
            torch.save(model.state_dict(), mode_weights_name)
            torch.save(model, model_save_name)
            
            # model_save_name = str(save_path + model_name + '/checkpoint.pth')
            # mode_weights_name = str(save_path + model_name + '/weights.pth')

            # torch.save(model.state_dict(), mode_weights_name)
            # torch.save(model, model_save_name)

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping')
            break

        scheduler.step()

    print("Finished Training Teacher")
    plot_loss_curve(val_losses)
    return model



# Function to train the student model with knowledge distillation
def train_student_with_distillation(student, teacher, trainloader, criterion, optimizer, scheduler, device, alpha, temperature, num_epochs, patience=5):
    
    best_val_loss = float('inf')
    patience_counter = 0
    epoch_losses = [] 
    val_losses = []
    
    student.train()
    teacher.eval()
    student.to(device)
    teacher.to(device)
    best_train_loss = float('inf')  
    patience_counter = 0 

    for epoch in range(num_epochs):
        running_loss = 0.0 
        epoch_loss = 0.0  
        num_batches = 0  
        # for i, batch in enumerate(tqdm(trainloader)):
        #     inputs, labels = batch['img'].to(device), batch['label'].to(device)
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            student_outputs = student(inputs)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            ce_loss = criterion(student_outputs[0], labels)
            kd_loss = tkd_kdloss(student_outputs[0], teacher_outputs[0], temperature=temperature)  # from utils.loss_functions
            loss = alpha * kd_loss + (1 - alpha) * ce_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            if i % 100 == 99:  
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        epoch_loss /= num_batches  

        epoch_losses.append(epoch_loss)

        
        model.eval()
        total_correct = 0
        total_samples = 0
        total_val_loss = 0.0
        num_batches = 0  
        with torch.no_grad():
            for inputs, labels in tqdm(testloader):
                val_inputs, val_labels = inputs.to(device), labels.to(device)
                # val_inputs = val_data['img'].to(device)
                # val_labels = val_data['label'].to(device)
    
                # Forward pass for validation
                _, val_outputs = model(val_inputs)
    
                val_loss = criterion(val_outputs, val_labels)

                total_val_loss += val_loss.item()
    
                # Compute the validation accuracy
                _, predicted = torch.max(val_outputs, 1)
                total_samples += val_labels.size(0)
                total_correct += (predicted == val_labels).sum().item()
                num_batches += 1
            total_val_loss /= num_batches
            val_losses.append(total_val_loss)
            accuracy = total_correct / total_samples
            print(f'*****Epoch {epoch + 1}/{num_epochs}*****\n' 
            f'*****Train Loss: {epoch_loss: .6f} Val Loss: {total_val_loss: .6f}*****\n'
            f'*****Validation Accuracy: {accuracy * 100:.2f}%*****\n')

        model.eval()
        total_correct = 0
        total_samples = 0
        total_val_loss = 0.0
        num_batches = 0  
        with torch.no_grad():
            for inputs, labels in tqdm(testloader):
                val_inputs, val_labels = inputs.to(device), labels.to(device)
                # val_inputs = val_data['img'].to(device)
                # val_labels = val_data['label'].to(device)
    
                # Forward pass for validation
                _, val_outputs = model(val_inputs)
    
                val_loss = criterion(val_outputs, val_labels)

                total_val_loss += val_loss.item()
    
                # Compute the validation accuracy
                _, predicted = torch.max(val_outputs, 1)
                total_samples += val_labels.size(0)
                total_correct += (predicted == val_labels).sum().item()
                num_batches += 1
            total_val_loss /= num_batches
            val_losses.append(total_val_loss)
            accuracy = total_correct / total_samples
            print(f'*****Epoch {epoch + 1}/{num_epochs}*****\n' 
            f'*****Train Loss: {epoch_loss: .6f} Val Loss: {total_val_loss: .6f}*****\n'
            f'*****Validation Accuracy: {accuracy * 100:.2f}%*****\n')

        

        # Check for early stopping
        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            patience_counter = 0 
            torch.save(student.state_dict(), f'student_model_weights_ckd_prof_checkpoint.pth')
            torch.save(student, f'student_model_ckd_prof_checkpoint.pth')
        else:
            patience_counter += 1 

        if patience_counter >= patience:
            print('Early stopping')
            break  

        scheduler.step() 


    print("Finished Training Student")
    plot_loss_curve(val_losses)
    
    return model

######################### WIDER STARTS

def best_LR_wider(save_name, model, dataloader, criterion, optimizer, scheduler, device, num_epochs=3, lr_range=(1e-4, 1e-1), plot_loss=True):

    model.train()
    model.to(device)
    lr_values = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), num_epochs * len(dataloader))  # Generate learning rates for each batch
    lr_iter = iter(lr_values)
    losses = []
    lrs = []
            
    for epoch in range(num_epochs):
        # for i, (inputs, labels) in enumerate(tqdm(dataloader)):
        #     lr = next(lr_iter)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr  # Set new learning rate
            
        #     inputs, labels = batch['img'].to(device), batch['label'].to(device)
        for i, batch in enumerate(tqdm(dataloader)):
            lr = next(lr_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr  # Set new learning rate
            
            inputs, labels = batch['img'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(type(outputs), outputs[0], outputs[1])
            # loss = criterion(outputs[0], labels)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            lrs.append(lr)
    
    # Calculate the derivative of the loss
    loss_derivative = np.gradient(losses)
    
    # Find the learning rate corresponding to the minimum derivative (steepest decline)
    best_lr_index = np.argmin(loss_derivative)
    best_lr = lrs[best_lr_index]
    
    if plot_loss:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Range Test')
        plt.axvline(x=best_lr, color='red', linestyle='--', label=f'Best LR: {best_lr}')
        plt.legend()
        plt.show()
    
    print(f'Best learning rate: {best_lr}')
    return best_lr



def train_teacher_wider(model_name, model, trainloader, criterion, optimizer, scheduler, num_epochs=240, patience=5):
    ''' A function to train the teacher models'''
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    best_train_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0  
        num_batches = 0  
        for i, batch in enumerate(tqdm(trainloader)):
            inputs, labels = batch['img'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        epoch_loss /= num_batches  
        
        # Check for early stopping
        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            patience_counter = 0 
            
            # checkpoint
            save_path = './weights/'

            model_save_path = os.path.join(save_path, model_name)
            
            os.makedirs(model_save_path, exist_ok=True)
        
            model_save_name = os.path.join(model_save_path, 'checkpoint.pth')
            mode_weights_name = os.path.join(model_save_path, 'weights.pth')
        
            torch.save(model.state_dict(), mode_weights_name)
            torch.save(model, model_save_name)
            
            # model_save_name = str(save_path + model_name + '/checkpoint.pth')
            # mode_weights_name = str(save_path + model_name + '/weights.pth')

            # torch.save(model.state_dict(), mode_weights_name)
            # torch.save(model, model_save_name)

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping')
            break

        scheduler.step()

    print("Finished Training Teacher")
    return model

######################### WIDER ENDS

#### Norm and Direction code helper functions ##

def get_emb_fea(model, dataloader, batch_size, emb_size = 64):
    ''' Used to extract the feature embeddings in a teacher model '''
    # model to evaluate mode
    model.eval()
    # dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
    dataloader = dataloader

    EMB = {}

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()

            # compute output
            emb_fea, logits = model(images, embed=True)

            for emb, i in zip(emb_fea, labels):
                i = i.item()
                assert len(emb) == emb_size
                if str(i) in EMB:
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))
                else:
                    EMB[str(i)] = [[] for _ in range(len(emb))]
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))


    for key, value in EMB.items():
        for i in range(emb_size):
            EMB[key][i] = round(np.array(EMB[key][i]).mean(), 4)

    return EMB



def retrieve_teacher_class_weights(model_name, model_weight_path, num_class, data_name, dataloader, batch_size):
    ''' Use the extracted feature embeddings to create a json of class means for teacher'''
    model = models_package.__dict__[model_name](num_class=num_class)
    model_ckpt = models_package.__dict__[model_name](num_class=num_class)
    print('Visualized the embedding feature of the {} model on the train set'.format(model_name))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ckpt.to(device)
    model_ckpt.load_state_dict(torch.load(model_weight_path))
    model_ckpt.eval()
    new_state_dict = OrderedDict()
    for k, v in model_ckpt.items():
        name = k[7:]   # remove 'module.'
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    for param in model.parameters():
        param.requires_grad = False
    
    model = model.cuda()

    emb = get_emb_fea(model=model, dataloader=dataloader, batch_size=batch_size)
    emb_json = json.dumps(emb, indent=4)
    with open("./class_means/{}_embedding_fea/{}.json".format(data_name, model_name), 'w', encoding='utf-8') as f:
        f.write(emb_json)
    f.close()


def new_teacher_class_weights(model_name, model_weight_path, num_class, data_name, dataloader, batch_size):
    ''' Use the extracted feature embeddings to create a json of class means for teacher'''
    model = models_package.__dict__[model_name](num_class=num_class)
    checkpoint=torch.load(model_weight_path)
    
    print('Visualized the embedding feature of the {} model on the train set'.format(model_name))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_ckpt.to(device)
    # model_ckpt.load_state_dict(torch.load(model_weight_path))
    # model_ckpt.eval()
    
    new_checkpoint = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:] # remove module.
        new_checkpoint[name] = v
    model.load_state_dict(new_checkpoint)

    for param in model.parameters():
        param.requires_grad = False
    
    model = model.cuda()

    emb = get_emb_fea(model=model, dataloader=dataloader, batch_size=batch_size)
    emb_json = json.dumps(emb, indent=4)
    with open("./class_means/{}_embedding_fea/{}.json".format(data_name, model_name), 'w', encoding='utf-8') as f:
        f.write(emb_json)
    f.close()


## Training script

def train_kd(model, teacher, T_EMB, train_dataloader, optimizer, criterion, kd_loss, nd_loss, args, epoch):
    train_loss = AverageMeter()
    train_error = AverageMeter()

    Cls_loss = AverageMeter()
    Div_loss = AverageMeter()
    Norm_Dir_loss = AverageMeter()

    # Model on train mode
    model.train()
    teacher.eval()
    step_per_epoch = len(train_dataloader)

    for step, (images, labels) in enumerate(train_dataloader):
        start = time.time()
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda() 

            # compute output
            s_emb, s_logits = model(images, embed=True)
    
            with torch.no_grad():
                t_emb, t_logits = teacher(images, embed=True)
    
            # cls loss
            cls_loss = criterion(s_logits, labels) * args.cls_loss_factor
            # KD loss
            div_loss = kd_loss(s_out = s_logits, t_out = t_logits) * min(1.0, epoch/args.warm_up)
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


def test_kd(model, test_dataloader, criterion):
    test_loss = AverageMeter()
    test_error = AverageMeter()

    # Model on eval mode
    model.eval()

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.cuda(), labels.cuda()

            # compute logits
            logits = model(images, embed=False)

            loss = criterion(logits, labels)

            # measure accuracy and record loss
            batch_size = images.size(0)
            _, pred = logits.data.cpu().topk(1, dim=1)
            test_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
            test_loss.update(loss.item(), batch_size)

    return test_loss.avg, test_error.avg


def epoch_loop_kd(model, teacher, train_loader, test_loader, num_class, args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = nn.DataParallel(model, device_ids=args.gpus)
    model = nn.DataParallel(model)
    model.to(device)
    # teacher = nn.DataParallel(teacher, device_ids=args.gpus)
    teacher = nn.DataParallel(teacher)
    teacher.to(device)

    # loss
    criterion = nn.CrossEntropyLoss().to(device)
    kd_loss = KDLoss(kl_loss_factor=args.kd_loss_factor, T=args.t).to(device)
    nd_loss = DirectNormLoss(num_class=num_class, nd_loss_factor=args.nd_loss_factor).to(device)
    # optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

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
        model.load_state_dict(checkpoint['model_state_dict'])
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

    # Train model
    best_error = 1
    for epoch in range(start_epoch, args.epochs):
        if epoch in [150, 180, 210]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        norm_dir_loss, div_loss, cls_loss, train_epoch_loss, train_error = train_kd(model=model,
                                                                                 teacher=teacher,
                                                                                 T_EMB=T_EMB,
                                                                                 train_dataloader=train_loader,
                                                                                 optimizer=optimizer,
                                                                                 criterion=criterion,
                                                                                 kd_loss=kd_loss,
                                                                                 nd_loss=nd_loss,
                                                                                 args=args,
                                                                                 epoch=epoch)
        test_epoch_loss, test_error = test_kd(model=model,
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

        # save model
        is_best = test_error < best_error
        best_error = min(best_error, test_error)
        state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
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
            summary_writer.add_image('Image', img_grid)
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        summary_writer.add_scalar('train_loss', train_epoch_loss, epoch)
        summary_writer.add_scalar('train_error', train_error, epoch)
        summary_writer.add_scalar('val_loss', test_epoch_loss, epoch)
        summary_writer.add_scalar('val_error', test_error, epoch)

        summary_writer.add_scalar('nd_loss', norm_dir_loss, epoch)
        summary_writer.add_scalar('kd_loss', div_loss, epoch)
        summary_writer.add_scalar('cls_loss', cls_loss, epoch)

    summary_writer.close()
    import os
    if not os.path.exists(train_acc_savepath) or not os.path.exists(train_loss_savepath):
        np.save(train_acc_savepath, train_acc)
        np.save(train_loss_savepath, train_loss)
        np.save(val_acc_savepath, test_acc)
        np.save(val_loss_savepath, test_loss)


######## ReviewKD++ ######
def train_reviewkd(model, teacher, T_EMB, train_dataloader, optimizer, criterion, nd_loss, args, epoch):
    train_loss = AverageMeter()
    train_error = AverageMeter()

    Cls_loss = AverageMeter()
    Div_loss = AverageMeter()
    Norm_Dir_loss = AverageMeter()

    # Model on train mode
    model.train()
    teacher.eval()
    step_per_epoch = len(train_dataloader)

    # pdb.set_trace()
    for step, (images, labels) in enumerate(train_dataloader):
        start = time.time()
        images, labels = images.cuda(), labels.cuda()

        # compute output
        s_features, s_emb, s_logits = model(images)

        with torch.no_grad():
            t_features, t_emb, t_logits = teacher(images, is_feat=True, preact=True)
            t_features = t_features[1:]

        # cls loss
        cls_loss = criterion(s_logits, labels) * args.cls_loss_factor
        # Kd loss
        kd_loss = hcl(s_features, t_features) * min(1, epoch/args.warm_up) * args.kd_loss_factor
        # ND loss
        norm_dir_loss = nd_loss(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels)

        loss = cls_loss + kd_loss + norm_dir_loss
        # measure accuracy and record loss
        batch_size = images.size(0)
        _, pred = s_logits.data.cpu().topk(1, dim=1)
        train_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        train_loss.update(loss.item(), batch_size)

        Cls_loss.update(cls_loss.item(), batch_size)
        Div_loss.update(kd_loss.item(), batch_size)
        Norm_Dir_loss.update(norm_dir_loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        s1 = '\r{} [{}/{}]'.format(t, step+1, step_per_epoch)
        s2 = ' - {:.2f}ms/step - nd_loss: {:.3f} - div_loss: {:.3f} - cls_loss: {:.3f} - train_loss: {:.3f} - train_acc: {:.3f}'.format(
             1000 * (time.time() - start), norm_dir_loss.item(), kd_loss.item(), cls_loss.item(), train_loss.val, 1-train_error.val)

        print(s1+s2, end='', flush=True)

    print()
    return Norm_Dir_loss.avg, Div_loss.avg, Cls_loss.avg, train_loss.avg, train_error.avg


def test_reviewkd(model, test_dataloader, criterion):
    test_loss = AverageMeter()
    test_error = AverageMeter()

    # Model on eval mode
    model.eval()

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.cuda(), labels.cuda()

            # compute logits
            _, _, logits = model(images)

            loss = criterion(logits, labels)

            # measure accuracy and record loss
            batch_size = images.size(0)
            _, pred = logits.data.cpu().topk(1, dim=1)
            test_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
            test_loss.update(loss.item(), batch_size)

    return test_loss.avg, test_error.avg


def epoch_loop_reviewkd(model, teacher, train_loader, test_loader, num_class, T_EMB, args):

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = nn.DataParallel(model, device_ids=args.gpus)
    model = nn.DataParallel(model)
    model.to(device)
    # teacher = nn.DataParallel(teacher, device_ids=args.gpus)
    teacher = nn.DataParallel(teacher)
    teacher.to(device)

    # loss
    criterion = nn.CrossEntropyLoss().to(device)
    nd_loss = DirectNormLoss(num_class=num_class, nd_loss_factor=args.nd_loss_factor).to(device)
    # optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # 权重
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
        model.load_state_dict(checkpoint['model_state_dict'])
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

    # Train model
    best_error = 1
    for epoch in range(start_epoch, args.epochs):
        if epoch in [150, 180, 210]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        norm_dir_loss, div_loss, cls_loss, train_epoch_loss, train_error = train_reviewkd(model=model,
                                                                                 teacher=teacher,
                                                                                 T_EMB=T_EMB,
                                                                                 train_dataloader=train_loader,
                                                                                 optimizer=optimizer,
                                                                                 criterion=criterion,
                                                                                 nd_loss=nd_loss,
                                                                                 args=args,
                                                                                 epoch=epoch)
        test_epoch_loss, test_error = test_reviewkd(model=model,
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

        # save model
        is_best = test_error < best_error
        best_error = min(best_error, test_error)
        state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
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
        # pdb.set_trace()
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

    summary_writer.close()
    if not os.path.exists(train_acc_savepath) or not os.path.exists(train_loss_savepath):
        np.save(train_acc_savepath, train_acc)
        np.save(train_loss_savepath, train_loss)
        np.save(val_acc_savepath, test_acc)
        np.save(val_loss_savepath, test_loss)


    

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def Save_Checkpoint(state, last, last_path, best, best_path, is_best):
    if os.path.exists(last):
        shutil.rmtree(last)
    last_path.mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(last_path, 'ckpt.pth'))

    if is_best:
        if os.path.exists(best):
            shutil.rmtree(best)
        best_path.mkdir(parents=True, exist_ok=True)
        torch.save(state, os.path.join(best_path, 'ckpt.pth'))

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


################## disparity sh*t######################

