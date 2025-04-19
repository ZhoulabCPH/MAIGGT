# MAIGGT
Code for 'An explainable multimodal artificial intelligence model integrating histopathological microenvironment and EHR phenotypes for germline genetic testing in breast cancer'
![schematic](https://github.com/ZhoulabCPH/MAIGGT/blob/master/checkpoints/schematic.png)
****
## Abstract
Genetic testing for pathogenic germline variants is critical for the personalised management of high-risk breast cancers, guiding targeted therapies and cascade testing for at-risk families. In this study, we propose MAIGGT (Multimodal Artificial Intelligence Germline Genetic Testing), a deep learning framework that integrates  histopathological microenvironment features from whole-slide images with clinical phenotypes from electronic health records for precise prescreening of germline BRCA1/2 mutations. Leveraging a multi-scale Transformer-based deep generative architecture, MAIGGT employs a cross-modal latent representation unification mechanism to capture complementary biological insights from multimodal data. MAIGGT was rigorously validated across three independent cohorts and demonstrated robust performance with areas under receiver operating characteristic curves of 0.925 (95% CI 0.868–0.982), 0.845 (95% CI 0.779–0.911) and 0.833 (0.788-0.878), outperforming single-modality models. Mechanistic interpretability analysis revealed that BRCA1/2-associated tumors exhibited distinct microenvironmental patterns, including increased inflammatory cell infiltration, stromal proliferation and necrosis, and nuclear heterogeneity. By bridging digital pathology with clinical phenotypes, MAIGGT establishes a new paradigm for cost-effective, scalable, and biologically interpretable prescreening of hereditary breast cancer, with the potential to significantly improve the accessibility of genetic testing in routine clinical practice.
****
## Dataset
- clinical_data: Path to the patient phenotype data.
- WSIs: Path to the patient's H&E-stained whole slide images. 
- Patches_224: Path to the patches of WSIs.
- Patches_512: Path to the patches of WSIs.
- WSIs_tumor_segmentation_224: Path to the tumor segmentation results of WSIs.
- WSIs_tumor_segmentation_512: Path to the tumor segmentation results of WSIs.
- WSIs_CTransPath_224: Path to the pathological feature representation of CTransPath for patches.
- WSIs_CTransPath_512: Path to the pathological feature representation of CTransPath for patches.
- WSIs_CTransPath_cluster_sample_224: Path to the cluster sampling results of patches.
- WSIs_CTransPath_cluster_sample_512: Path to the cluster sampling results of patches. 
  The datasets are available from the corresponding author upon reasonable request.

## checkpoints
- CTransPath: CTransPath model pretrained by [CTransPath](https://github.com/Xiyue-Wang/TransPath).
- tumor_segmentation_model_224: tumor segmentation model on patch of size 224.
- tumor_segmentation_model_512: tumor segmentation model on patch of size 512.
- WISE-BRCA: Whole-slide Images Systematically Extrapolate BRCA1/2 mutations.
- WISE-BRCA-CNB: Whole-slide Images Systematically Extrapolate BRCA1/2 mutations on biopsy samples.
- MAIGGT_mcVAE: MAIGGT_mcVAE is used to lean a joint common latent space of heterogeneous histopathological and phenotypic data.
- MAIGGT: Joint prediction of BRCA1/2 mutation carriers from histology images and electronic health records.
All checkpoints can be found at [MAIGGT](https://drive.google.com/drive/folders/1g4M8utv8-lPsp0yvJKDFEXheYQ6gPEti?usp=sharing).
## data_preprocessing
- <code>tiling_WSI_multi_thread.py</code>: Used to segment and filter patches from WSIs. Implemented based on <code>histolab</code> package.
- <code>stain_normalization_multi_thread.py</code>: Patches stain normalization. Implemented based on <code>ParamNet</code>.
- <code>cluster_sample.py</code>: A clustering-based sampling strategy implement to extract patches with distinct histomorphological features from the tumor area.
- <code>phenotype_data_preprocessing.py</code>: Preprocessing pipeline of patient phenotype data.

## tumor_segmentation
- <code>dataset.py</code>: Generate datasets.
- <code>model.py</code>: Implementation of tumor segmentation model.
- <code>train.py</code>: Training the tumor segmentation model.
- <code>inference_to_datasets.py</code>: Using tumor segmentation model to automatically extract tumor areas from each WSI.

## get_patches_feature
- <code>ctran.py</code>: Implementation of CTransPath.
- <code>get_CTransPath_feature.py</code>: Using pre-trained CTransPath to obtain histopathological features of patches.
  
  Part of the implementation here is based on [CTransPath](https://github.com/Xiyue-Wang/TransPath).

## WISE-BRCA
- <code>dataset.py</code>: Generate datasets.
- <code>model.py</code>: Implementation of WISE-BRCA.
- <code>train.py</code>: Training the WISE-BRCA.
- <code>inference.py</code>: Predicting germline BRCA1/2 mutation status from histology images using WISE-BRCA.

## MAIGGT
- MAIGGT_mcVAE
  - <code>dataset.py</code>: Generate datasets.
  - <code>model_MAIGGT_mcVAE.py</code>: Implementation of MAIGGT_mcVAE.
  - <code>model_WISE-BRCA.py</code>: Implementation of WISE-BRCA.
  - <code>train.py</code>: Training the MAIGGT_mcVAE.

- <code>model.py</code>: Implementation of MAIGGT.
- <code>train.py</code>: Training the MAIGGT.
- <code>inference.py</code>: Predicting germline BRCA1/2 mutation status from histology images and electronic health records using MAIGGT.

## Usage tutorial
If you intend to utilize it for paper reproduction or your own dataset, please adhere to the following workflow:
  1) Configuration Environment.
  2) Prepare dataset. Prepare your own dataset in the path <code>datasets</code>. Please store the WSIs (Whole Slide Images) in the <code>datasets/WSIs</code> folder. Additionally, if available, store the phenotypic data of the corresponding patients in the <code>datasets/clinical_data/phenotype_data_original.csv</code> file (refer to the sample data in the table). Here we provide two samples, whose original phenotypic data and secondary data have been stored in <code>datasets</code>. The original WSI can be download from the [Google Drive](https://drive.google.com/drive/folders/1OA2Dp_P82qsCn4yOi_r33qgsYilfFlf6).
  3) Use <code>data_preprocessing/tiling_WSI_multi_thread.py</code> to segment WSIs into patches of size 224 and 512 at mpp of 0.488.
  4) Use <code>data_preprocessing/stain_normalization_multi_thread.py</code> to perform stain normalization for patches (If computing resources are limited, consider applying stain normalization only to patches sampled from the cluster sample).
  5) Use <code>get_patches_feature/get_CTransPath_feature.py</code> to obtain representation vector of patches.
  6) Use <code>tumor_segmentation/inference_to_datasets.py</code> to extract tumor areas from each WSI.
  7) Use <code>data_preprocessing/cluster_sample.py</code> clustering-based sampling strategy to extract patches with distinct histomorphological features from the tumor area.
  8) Use <code>data_preprocessing/phenotype_data_preprocessing.py</code> to process the phenotype data. For more details, please refer to our previously published work 'DrABC: deep learning accurately predicts germline pathogenic mutation status in breast cancer patients based on phenotype data'.
  9) After preparation, use <code>WISE-BRCA/inference.py</code> or <code>MAIGGT/inference.py</code> to predict germline BRCA1/2 mutation status on your own datasets. Or use <code>WISE-BRCA/train.py</code> or <code>MAIGGT/train.py</code> on your own datasets.
```text
Note: The above code uses relative paths by default, and all results will be automatically saved accordingly. You may modify the paths to store the results in any desired location.
```
  






  





  
