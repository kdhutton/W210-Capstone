import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict
import shutil, os
import json


def get_emb_fea(model, dataloader):
    ''' Used to extract the feature embeddings in a teacher model '''
    
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    
    model.eval()
    model.avgpool.register_forward_hook(get_features('feats'))

    EMB = {}


    with torch.no_grad():

        for batch_data in dataloader:
            FEATS = []
            features = {}
            images = batch_data["img"].to(device)
            labels = batch_data["label"].to(device)
            curr_batch_size = len(images)

            # compute output
            outputs = model(images)
            FEATS.append(features['feats'].cpu().numpy())
            emb_fea = np.concatenate(FEATS)
            # reshape embedding features to flatten 
            emb_fea = emb_fea.reshape((curr_batch_size, emb_fea.shape[1]))

            for emb, i in zip(emb_fea, labels):
                i = i.item()
                emb_size = len(emb) 
                if str(i) in EMB:
                    for j in range(emb_size):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))
                else:
                    EMB[str(i)] = [[] for _ in range(emb_size)]
                    for j in range(emb_size):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))

    for key, value in EMB.items():
        for i in range(emb_size):
            EMB[key][i] = round(np.array(EMB[key][i]).mean(), 4)

    return EMB

def retrieve_teacher_class_weights(model, lmda, data_name, dataloader, model_name, best_model_weights):
    ''' Use the extracted feature embeddings to create a json of class means for teacher'''

    checkpoint = best_model_weights

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    
    model = model.cuda()

    emb = get_emb_fea(model=model, dataloader=dataloader)
    emb_json = json.dumps(emb, indent=4)

    # Create the directory if it doesn't exist
    output_dir = "./class_means/{}_embedding_fea".format(data_name)
    os.makedirs(output_dir, exist_ok=True)

    with open("{}/{}_lmda_{}.json".format(output_dir, model_name, lmda), 'w', encoding='utf-8') as f:
        f.write(emb_json)