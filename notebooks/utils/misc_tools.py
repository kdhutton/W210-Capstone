import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import models_package
from collections import OrderedDict


#### finding the optimal learning rate
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



def train_teacher(model_name, model, trainloader, criterion, optimizer, scheduler, num_epochs=240, patience=5):
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
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        epoch_loss /= num_batches  
        
        # Check for early stopping
        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            patience_counter = 0 
            # checkpoint
            save_path = './weights/'
            model_save_name = str(save_path + model_name + '/checkpoint.pth')
            mode_weights_name = str(save_path + model_name + '/weights.pth')

            torch.save(model.state_dict(), mode_weights_name)
            torch.save(model, model_save_name)

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping')
            break

        scheduler.step()

    print("Finished Training Teacher")
    return model


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
