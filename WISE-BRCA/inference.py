import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np

from model import WISE_BRCA
from datasets import BC, collate
from torch.utils.data import DataLoader
from torch import nn



def pgBRCA_mutation_predict():
    slides_path_224 = rf'../datasets/WSIs_CTransPath_cluster_sample_224'
    slides_path_512 = rf'../datasets/WSIs_CTransPath_cluster_sample_512'

    model = WISE_BRCA().cuda()
    ckpt = torch.load(
        rf'../checkpoints/WISE-BRCA.pth',
        map_location='cuda:0')
    model.load_state_dict(ckpt['model'])

    slides_name = os.listdir(slides_path_224)
    labels = [0 for i in range(len(slides_name))]

    workspace = pd.DataFrame()
    workspace['slides'] = slides_name
    workspace['BRCA_mut'] = labels

    data_inference = BC(workspace, slides_path_224, slides_path_512)
    data_inference_loader = DataLoader(data_inference, 16, shuffle=False, num_workers=4, drop_last=False,
                                   collate_fn=collate)
    with torch.no_grad():
        sigmoid = nn.Sigmoid()
        data = pd.DataFrame()
        model.eval()
        sample = []
        WISE_BRCA_score = np.array([])
        label = []
        for step, slide in enumerate(data_inference_loader):
            patches_224, patches_name_224, patches_512, patches_name_512, batch_labels, sample_ = slide[
                'patches_features_224'], slide['patches_names_224'], slide['patches_features_512'], slide[
                'patches_names_512'], slide['labels'], slide['Sample']
            sample = sample + sample_
            pred_ = model.forward(patches_224.cuda(), patches_512.cuda())

            WISE_BRCA_score = np.append(WISE_BRCA_score, (sigmoid(pred_).detach().cpu().numpy()))

            label = label + list(batch_labels.detach().cpu().numpy())

        data['sample'] = list(sample)
        data['WISE_BRCA_score'] = list(WISE_BRCA_score)

    return data




if __name__ == '__main__':
    pgBRCA_mutation_predict()














