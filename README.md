# Improving Image Quality of Sparse-view Lung Cancer CT Images with  U-Net

Code to the paper: "Improving Image Quality of Sparse-view Lung Cancer CT Images with  U-Net."

## Abstract

- **Background:** We aimed at improving image quality (IQ) of sparse-view computed tomography (CT) images using a U-Net for lung metastasis detection and determining the best tradeoff between number of views, IQ, and diagnostic confidence.

- **Methods:** CT images from 41 subjects aged 62.8 ± 10.6 years (mean ± standard deviation), 23 men, 34 with lung metastasis, 7 healthy, were retrospectively selected (2016–2018) and forward projected onto 2,048-view sinograms. Six corresponding sparse-view CT data subsets at varying levels of undersampling were reconstructed from sinograms using filtered backprojection with 16, 32, 64, 128, 256, and 512 views. A dual-frame U-Net was trained and evaluated for each subsampling level on 8,658 images from 22 diseased subjects. A representative image per scan was selected from 19 subjects (12 diseased, 7 healthy) for a single-blinded multireader study. These slices, for all levels of subsampling, with and without U-Net postprocessing, were presented to three readers. IQ and diagnostic confidence were ranked using predefined scales. Subjective nodule segmentation was evaluated using sensitivity and Dice similarity coefficient (DSC); clustered Wilcoxon signed-rank test was used.

- **Results:** The 64-projection sparse-view images resulted in 0.89 sensitivity and 0.81 DSC, while their counterparts, postprocessed with the U-Net, had improved metrics (0.94 sensitivity and 0.85 DSC) (p = 0.400). Fewer views led to insufficient IQ for diagnosis. For increased views, no substantial discrepancies were noted between sparse-view and postprocessed images.

- **Conclusion:** Projection views can be reduced from 2,048 to 64 while maintaining IQ and the confidence of the radiologists on a satisfactory level.



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

 
------------------------
 Authored by:
- Annika Ries (annika97.ries@gmail.com)
- Tina Dorosti (tina.dorosti@tum.de)
