import pandas as pd
import numpy as np

def preprocessing_phenotype_data():
    phenotype_data_original = pd.read_csv(rf'../datasets/clinical_data/phenotype_data_original.csv')
    age_at_diagnosis = phenotype_data_original.loc[:, 'age_at_diagnosis'].to_list()
    if True in np.isnan(age_at_diagnosis):
        print('Exist NaN')
    age_at_diagnosis = [item / 100 for item in age_at_diagnosis]
    phenotype_data_original['age_at_diagnosis'] = age_at_diagnosis

    # tumor_history
    tumor_history = phenotype_data_original.loc[:, 'tumor_history'].to_list()
    if True in np.isnan(tumor_history):
        print('Exist NaN')
    phenotype_data_original['tumor_history'] = tumor_history

    # BRCA_history
    BRCA_history = phenotype_data_original.loc[:, 'BRCA_history'].to_list()
    if True in np.isnan(BRCA_history):
        print('Exist NaN')
    phenotype_data_original['BRCA_history'] = BRCA_history

    # OV_history
    OV_history = phenotype_data_original.loc[:, 'OV_history'].to_list()
    if True in np.isnan(OV_history):
        print('Exist NaN')
    phenotype_data_original['OV_history'] = OV_history

    # tumor_family_history
    tumor_family_history = phenotype_data_original.loc[:, 'tumor_family_history'].to_list()
    if True in np.isnan(tumor_family_history):
        print('Exist NaN')
    phenotype_data_original['tumor_family_history'] = tumor_family_history

    # BRCA_family_history
    BRCA_family_history = phenotype_data_original.loc[:, 'BRCA_family_history'].to_list()
    if True in np.isnan(BRCA_family_history):
        print('Exist NaN')
    phenotype_data_original['BRCA_family_history'] = BRCA_family_history

    # OV_family_history
    OV_family_history = phenotype_data_original.loc[:, 'OV_family_history'].to_list()
    if True in np.isnan(OV_family_history):
        print('Exist NaN')
    phenotype_data_original['OV_family_history'] = OV_family_history

    # pancreatic_cancer_family_history
    pancreatic_cancer_family_history = phenotype_data_original.loc[:, 'pancreatic_cancer_family_history'].to_list()
    if True in np.isnan(pancreatic_cancer_family_history):
        print('Exist NaN')
    phenotype_data_original['pancreatic_cancer_family_history'] = pancreatic_cancer_family_history

    # mbc_cancer_family_history (Man breast cancer family history)
    mbc_cancer_family_history = phenotype_data_original.loc[:, 'mbc_cancer_family_history'].to_list()
    if True in np.isnan(mbc_cancer_family_history):
        print('Exist NaN')
    phenotype_data_original['mbc_cancer_family_history'] = mbc_cancer_family_history

    # largest_diameter
    largest_diameter = phenotype_data_original.loc[:, 'largest_diameter'].to_list()
    if True in np.isnan(largest_diameter):
        print('Exist NaN')
    largest_diameter = [item / 10 for item in largest_diameter]
    phenotype_data_original['largest_diameter'] = largest_diameter

    # Grade
    Grade = phenotype_data_original.loc[:, 'Grade'].to_list()
    if True in np.isnan(Grade):
        print('Exist NaN')
    Grade = [item / 3 for item in Grade]
    phenotype_data_original['Grade'] = Grade

    # AR_grade
    AR_grade = phenotype_data_original.loc[:, 'AR_grade'].to_list()
    if True in np.isnan(AR_grade):
        print('Exist NaN')
    AR_grade_ = []
    for ag in AR_grade:
        if type(ag) == float:
            if ag < 0.01:
                AR_grade_.append(0)
            elif ag <= 0.2:
                AR_grade_.append(1)
            elif ag <= 0.5:
                AR_grade_.append(2)
            elif ag > 0.5:
                AR_grade_.append(3)
        else:
            AR_grade_.append(ag)
    AR_grade_ = [item / 3 for item in AR_grade_]
    phenotype_data_original['AR_grade'] = AR_grade_

    # ER_grade
    ER_grade = phenotype_data_original.loc[:, 'ER_grade'].to_list()
    if True in np.isnan(AR_grade):
        print('Exist NaN')
    ER_grade_ = []
    for eg in ER_grade:
        if type(eg) == float:
            if eg < 0.01:
                ER_grade_.append(0)
            elif eg <= 0.2:
                ER_grade_.append(1)
            elif eg <= 0.5:
                ER_grade_.append(2)
            elif eg > 0.5:
                ER_grade_.append(3)
        else:
            ER_grade_.append(eg)
    ER_grade_ = [item / 3 for item in ER_grade_]
    phenotype_data_original['ER_grade'] = ER_grade_

    # PR_grade
    PR_grade = phenotype_data_original.loc[:, 'PR_grade'].to_list()
    if True in np.isnan(AR_grade):
        print('Exist NaN')
    PR_grade_ = []
    for pg in PR_grade:
        if type(pg) == float:
            if pg < 0.01:
                PR_grade_.append(0)
            elif pg <= 0.2:
                PR_grade_.append(1)
            elif pg <= 0.5:
                PR_grade_.append(2)
            elif pg > 0.5:
                PR_grade_.append(3)
        else:
            PR_grade_.append(pg)
    PR_grade_ = [item / 3 for item in PR_grade_]
    phenotype_data_original['PR_grade'] = PR_grade_

    # ki67
    ki67 = phenotype_data_original.loc[:, 'Ki67'].to_list()
    if True in np.isnan(ki67):
        print('Exist NaN')
    phenotype_data_original['ki67'] = ki67

    # CK5/6
    CK56 = phenotype_data_original.loc[:, 'CK56'].to_list()
    if phenotype_data_original['CK56'].isnull().any():
        print('Exist NaN')
    CK56 = [1 if item == 'Positive' else 0 for item in CK56]

    phenotype_data_original['CK56'] = CK56


    # Lymph_node

    Lymph_node_status = phenotype_data_original.loc[:, 'Lymph_node_status'].to_list()
    if phenotype_data_original['Lymph_node_status'].isnull().any():
        print('Exist NaN')
    Lymph_node_status = [1 if item == 'Invasion' else 0 for item in Lymph_node_status]
    phenotype_data_original['Lymph_node_status'] = Lymph_node_status

    # HER2_0
    HER2_0 = phenotype_data_original.loc[:, 'HER2'].to_list()
    if phenotype_data_original['HER2'].isnull().any():
        print('Exist NaN')


    HER2_0 = [0 if item == 'Positive' else 1 for item in HER2_0]
    phenotype_data_original['HER2_0'] = HER2_0

    # HER2_1
    HER2_1 = phenotype_data_original.loc[:, 'HER2'].to_list()
    if phenotype_data_original['HER2'].isnull().any():
        print('Exist NaN')
    HER2_1 = [1 if item == 'Positive' else 0 for item in HER2_1]
    phenotype_data_original['HER2_1'] = HER2_1

    # multifocal_1
    multifocal_1 = phenotype_data_original.loc[:, 'Multi-focal'].to_list()
    if True in np.isnan(multifocal_1):
        print('Exist NaN')
    phenotype_data_original['multifocal_1'] = multifocal_1

    # multifocal_2
    multifocal_2 = phenotype_data_original.loc[:, 'Bilateral breast lesions'].to_list()
    if True in np.isnan(multifocal_1):
        print('Exist NaN')
    phenotype_data_original['multifocal_2'] = multifocal_2

    # BRCA_mut
    BRCA_mut = phenotype_data_original.loc[:, 'BRCA_mut'].to_list()
    if True in np.isnan(BRCA_mut):
        print('Exist NaN')
    phenotype_data_original['BRCA_mut'] = BRCA_mut
    phenotype_data = phenotype_data_original.loc[:, ['Sample', 'BRCA_mut', 'age_at_diagnosis',
                                     'tumor_history', 'BRCA_history', 'OV_history', 'tumor_family_history',
                                     'BRCA_family_history', 'OV_family_history',
                                     'pancreatic_cancer_family_history', 'mbc_cancer_family_history',
                                     'largest_diameter', 'Grade', 'AR_grade', 'ER_grade', 'PR_grade', 'Ki67',
                                     'CK56', 'Lymph_node_status', 'HER2_0', 'HER2_1', 'multifocal_1',
                                     'multifocal_2']]
    phenotype_data.to_csv(rf'../datasets/clinical_data/phenotype_data.csv',
                       index=None)

preprocessing_phenotype_data()