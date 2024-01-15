# Improving Image Quality of Sparse-view Lung Cancer CT Images with  U-Net

Code to the paper: "Improving Image Quality of Sparse-view Lung Cancer CT Images with  U-Net."

## Abstract

- **Background:** To improve the image quality of sparse-view computed tomography (CT) images with a U-Net for lung cancer detection and to determine the best trade-off between number of views, image quality, and diagnostic confidence.
- **Methods:** CT images from 41 anonymized subjects (mean age, 62.8±10.6 years, 23 men; 34 with lung cancer, seven healthy) were retrospectively selected (01.2016-12.2018) and forward projected onto 2048-view sinograms. Six corresponding sparse-view CT data subsets at varying levels of undersampling were reconstructed from sinograms using filtered back projection with 16, 32, 64, 128, 256, and 512 views, respectively. A dual-frame U-Net was trained and evaluated for each subsampling level on 8,658 images from 22 diseased subjects. A representative image per scan was selected from 19 subjects (12 diseased, seven healthy) for a single-blinded reader study. The selected slices, for all levels of subsampling, with and without post-processing by the U-Net model, were presented to three readers. Image quality and diagnostic confidence were ranked using pre-defined scales. Subjective nodule segmentation was evaluated utilizing sensitivity and Dice Similarity Coefficient (DSC).
- **Results:** The 64-projection sparse-view images resulted in sensitivity=0.89 and DSC=0.81, while their counterparts, post-processed with the U-Net, had improved metrics (sensitivity = 0.94, DSC = 0.85 with *p* = 0.400). Fewer views led to insufficient quality for diagnostic purposes. For increased views, no substantial discrepancies were noted between the sparse-view and post-processed images.
- **Conclusion:** Projection views can be reduced from 2048 to 64 while maintaining image quality and the confidence of the radiologists on a satisfactory level.


## Getting Started

### Dependencies
- python==3.8.10
- tensorflow==2.4.0
- astra==2.1.1
- pandas==1.3.4
- scipy==1.4.1

### Executing program

- Obtain data
- Sparse-sample data with 1_dataPrep_sparseSampling.ipynb
- Train network with 2_train.ipynb
- Test model and obtain predicted images with 3_test.ipynb
- Evaluate results of reader study with 4_evalResults_readerStudy.ipynb