import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np

from model import MAIGGT
from MAIGGT_mcVAE.datasets import BC_multi_modal, collate_multi_modal
from torch.utils.data import DataLoader
from torch import nn



def pgBRCA_mutation_predict_multi_modal():
    slides_path_224 = rf'../datasets/WSIs_CTransPath_cluster_sample_224'
    slides_path_512 = rf'../datasets/WSIs_CTransPath_cluster_sample_512'
    phenotype_data_path = rf'../datasets/clinical_data/phenotype_data.csv'
    model = MAIGGT().cuda()
    ckpt = torch.load(
        rf'../checkpoints/MAIGGT.pth',
        map_location='cuda:0')
    model.load_state_dict(ckpt['model'])

    slides_name = os.listdir(slides_path_224)

    phenotype_data = pd.read_csv(phenotype_data_path)
    columns = ['Sample', 'BRCA_mut', 'age_at_diagnosis',
               'tumor_history', 'BRCA_history', 'OV_history', 'tumor_family_history',
               'BRCA_family_history', 'OV_family_history',
               'pancreatic_cancer_family_history', 'mbc_cancer_family_history',
               'largest_diameter', 'Grade', 'AR_grade', 'ER_grade', 'PR_grade', 'Ki67',
               'CK56', 'Lymph_node_status', 'HER2_0', 'HER2_1', 'multifocal_1',
               'multifocal_2']
    phenotype_data = phenotype_data.loc[:, columns]
    workspace= phenotype_data
    workspace.index = workspace.loc[:, 'Sample'].to_list()
    data_inference = BC_multi_modal(workspace, slides_path_224, slides_path_512)
    data_inference_loader = DataLoader(data_inference, 16, shuffle=False, num_workers=4, drop_last=False,
                                   collate_fn=collate_multi_modal)
    with torch.no_grad():
        sigmoid = nn.Sigmoid()
        report = pd.DataFrame()
        model.eval()
        sample = []
        MAIGGT_score = np.array([])
        label = []
        for step, slide in enumerate(data_inference_loader):
            patches_224, patches_name_224, patches_512, patches_name_512, phenotype_feature, batch_labels, sample_ = slide[
                'patches_features_224'], slide['patches_names_224'], slide['patches_features_512'], slide[
                'patches_names_512'], slide['phenotype_feature'], slide['labels'], slide['Sample']
            sample = sample + sample_
            pred_ = model.forward(patches_224.cuda(), patches_512.cuda(), phenotype_feature.cuda())

            MAIGGT_score = np.append(MAIGGT_score, (sigmoid(pred_).detach().cpu().numpy()))


        report['Sample'] = list(sample)
        report['MAIGGT_score'] = list(MAIGGT_score)
        cutpoint = 0.551622 # 80% Sensitivity
        for i in range(len(report)):
            sample = report.iloc[i,0]
            MAIGGT_score = report.iloc[i ,1]
            if MAIGGT_score >= cutpoint:
                result = 'Pathogenetic germline BRCA1/2 carrier'
            else:
                result = 'Non-carrier'
            print(rf'Sample {sample}, MAIGGT_score: {MAIGGT_score}, predict to {result}')
    return report


if __name__ == '__main__':
    pgBRCA_mutation_predict_multi_modal()

















