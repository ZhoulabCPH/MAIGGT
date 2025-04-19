import numpy as np
import pandas as pd
import feather
import torchvision.transforms as transforms
import torch
import os

from torch.utils.data import Dataset
from PIL import Image


def collate_multi_modal(batch):
    patches_features_224 = [b['patches_features_224'] for b in batch]
    patches_names_224 = [b['patches_names_224'] for b in batch]
    patches_features_512 = [b['patches_features_512'] for b in batch]
    patches_names_512 = [b['patches_names_512'] for b in batch]
    phenotype_feature = [b['phenotype_feature'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch])

    Sample = [b['Sample'] for b in batch]
    patches_features_224 = torch.tensor(patches_features_224, dtype=torch.float)
    patches_features_512 = torch.tensor(patches_features_512, dtype=torch.float)
    phenotype_feature = torch.tensor(phenotype_feature, dtype=torch.float)
    return {'patches_features_512': patches_features_512, 'patches_names_512': patches_names_512,
            'patches_features_224': patches_features_224, 'patches_names_224': patches_names_224,'labels': labels, 'Sample': Sample,
            'phenotype_feature': phenotype_feature}


class BC_multi_modal(Dataset):
    def __init__(self, workspace, slides_dir_224, slides_dir_512):
        super(BC_multi_modal, self).__init__()
        self.workspace = workspace
        self.slides_dir_224 = slides_dir_224
        self.slides_dir_512 = slides_dir_512
        self.slides = list(self.workspace.loc[:, 'Sample'])
    def __len__(self):
        return len(self.slides)

    def load_patches_features(self, slide_name):
        slide_name = rf'{slide_name}.feather'
        slide_dir_224 = f'{self.slides_dir_224}/{slide_name}'

        slide_224 = feather.read_dataframe(slide_dir_224)
        slide_dir_512 = f'{self.slides_dir_512}/{slide_name}'
        slide_512 = feather.read_dataframe(slide_dir_512)

        patches_names_224 = slide_224.index
        patches_names_512 = slide_512.index
        patches_features_224 = slide_224.iloc[:, 0:].values
        patches_features_512 = slide_512.iloc[:, 0:].values



        return patches_names_224, patches_features_224, patches_names_512, patches_features_512



    def __getitem__(self, item):
        label = self.workspace.iloc[item]['BRCA_mut']
        slide_name = self.workspace.iloc[item]['Sample']
        sample_name = slide_name.split('.')[0]
        columns = ['age_at_diagnosis',
               'tumor_history', 'BRCA_history', 'OV_history', 'tumor_family_history',
               'BRCA_family_history', 'OV_family_history',
               'pancreatic_cancer_family_history', 'mbc_cancer_family_history',
               'largest_diameter', 'Grade', 'AR_grade', 'ER_grade', 'PR_grade', 'Ki67',
               'CK56', 'Lymph_node_status', 'HER2_0', 'HER2_1', 'multifocal_1',
               'multifocal_2']


        phenotype_feature = self.workspace.loc[slide_name, columns].values
        Sample = sample_name

        patches_names_224, patches_features_224, patches_names_512, patches_features_512 = self.load_patches_features(slide_name)
        return {'patches_features_224': patches_features_224, 'patches_names_224': patches_names_224,
                'patches_features_512': patches_features_512, 'patches_names_512': patches_names_512,'label': label, 'Sample': Sample,
                'phenotype_feature': phenotype_feature}