# MAIGGT [![DOI](https://zenodo.org/badge/850167237.svg)](https://doi.org/10.5281/zenodo.14058685)
Code for 'An explainable multimodal artificial intelligence model integrating histopathological microenvironment and EHR phenotypes for germline genetic testing in breast cancer'
![schematic](https://github.com/ZhoulabCPH/WISE-BRCA/blob/master/checkpoints/schematic.png)
****
## Abstract
Genetic testing for pathogenic germline variants is critical for the personalised management of high-risk breast cancers, guiding targeted therapies and cascade testing for at-risk families. In this study, we propose MAIGGT (Multimodal Artificial Intelligence Germline Genetic Testing), a deep learning framework that integrates  histopathological microenvironment features from whole-slide images with clinical phenotypes from electronic health records for precise prescreening of germline BRCA1/2 mutations. Leveraging a multi-scale Transformer-based deep generative architecture, MAIGGT employs a cross-modal latent representation unification mechanism to capture complementary biological insights from multimodal data. MAIGGT was rigorously validated across three independent cohorts and demonstrated robust performance with areas under receiver operating characteristic curves of 0.925 (95% CI 0.868–0.982), 0.845 (95% CI 0.779–0.911) and 0.833 (0.788-0.878), outperforming single-modality models. Mechanistic interpretability analysis revealed that BRCA1/2-associated tumors exhibited distinct microenvironmental patterns, including increased inflammatory cell infiltration, stromal proliferation and necrosis, and nuclear heterogeneity. By bridging digital pathology with clinical phenotypes, MAIGGT establishes a new paradigm for cost-effective, scalable, and biologically interpretable prescreening of hereditary breast cancer, with the potential to significantly improve the accessibility of genetic testing in routine clinical practice.
****
## Dataset
- CHCAMS, Chinese Academy of Medical Sciences.
- YYH, Yantai Yuhuangding Hospital.
- HMUCH, Harbin Medical University Cancer Hospital.

  The datasets are available from the corresponding author upon reasonable request.

## checkpoints
- CTransPath: CTransPath model pretrained by [CTransPath](https://github.com/Xiyue-Wang/TransPath).
- Tumour_segmentation_model_224: Tumour segmentation model on patch of size 224.
- Tumour_segmentation_model_512: Tumour segmentation model on patch of size 512.
- WISE-BRCA: Whole-slide Images Systematically Extrapolate BRCA1/2 mutations.
- WISE-BRCA-biopsy: Whole-slide Images Systematically Extrapolate BRCA1/2 mutations on biopsy samples.
- MAIGGT_mcVAE: MAIGGT_mcVAE is used to lean a joint common latent space of heterogeneous histopathological and phenotypic data.
- MAIGGT: Joint prediction of BRCA1/2 mutation carriers from histology images and electronic health records.
All checkpoints can be found at [WISE-BRCA](https://drive.google.com/drive/folders/1g4M8utv8-lPsp0yvJKDFEXheYQ6gPEti?usp=sharing).
## data_preprocessing
- <code>tiling_WSI_multi_thread.py</code>: Used to segment and filter patches from WSIs. Implemented based on <code>histolab</code> package.
- <code>stain_normalization_multi_thread.py</code>: Patches stain normalization. Implemented based on <code>ParamNet</code>.
- <code>cluster_sample.py</code>: A clustering-based sampling strategy implement to extract patches with distinct histomorphological features from the tumour area.

## tumour_segmentation
- <code>dataset.py</code>: Generate datasets.
- <code>model.py</code>: Implementation of tumour segmentation model.
- <code>train.py</code>: Training the tumour segmentation model.
- <code>inference_to_datasets.py</code>: Using tumour segmentation model to automatically extract tumour areas from each WSI.

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

## Usage
If you intend to utilize it for paper reproduction or your own dataset, please adhere to the following workflow:
  1) Configuration Environment.
  2) Create a folder for your data in <code>datasets</code>.
  3) Use <code>data_preprocessing/tiling_WSI_multi_thread.py</code> to segment WSIs into patches of size 224 and 512 at mpp of 0.488.
  4) Use <code>data_preprocessing/stain_normalization_multi_thread.py</code> to perform stain normalization for patches (If computing resources are limited, consider applying stain normalization only to patches sampled from the cluster sample).
  5) Use <code>get_patches_feature/get_CTransPath_feature.py</code> to obtain representation vector of patches.
  6) Use <code>tumour_segmentation/inference_to_datasets.py</code> to extract tumour areas from each WSI.
  7) Use <code>data_preprocessing/cluster_sample.py</code> clustering-based sampling strategy to extract patches with distinct histomorphological features from the tumour area.
  8) For the processing of clinical information, please refer to our previously published work 'DrABC: deep learning accurately predicts germline pathogenic mutation status in breast cancer patients based on phenotype data'.
  9) After preparation, use <code>WISE-BRCA/inference.py</code> or <code>MAIGGT/inference.py</code> to predict germline BRCA1/2 mutation status on your own datasets. Or use <code>WISE-BRCA/train.py</code> or <code>MAIGGT/train.py</code> on your own datasets.
  






  





  
