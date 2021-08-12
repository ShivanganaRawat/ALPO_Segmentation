# How Useful is Image-Based Active Learning for Plant Organ Segmentation?

This is the official code implementation of our paper which is currently under a journal review.

## Datasets
|                 Dataset                |                                  Link                                  |
|:--------------------------------------:|:----------------------------------------------------------------------:|
| Apple (ACFR Orchard Fruit Dataset) [1] | [Download](https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit) |
|          UTokyo Wheat 2020 [2]         |                                  Soon                                  |
|          UTokyo Rice 2013 [3]          |                                  Soon                                  |


## Abstract
Training deep learning models typically requires a huge amount of labeled data which is expensive to acquire, especially in dense prediction tasks such as semantic segmentation. Moreover, plant phenotyping datasets pose additional challenges of heavy occlusion and varied lighting conditions which makes annotations more time-consuming to obtain. Active learning helps in reducing the annotation cost by selecting samples for labeling which are most informative to the model, thus improving model performance with fewer annotations. Active learning for semantic segmentation has been well studied on datasets such as PASCAL VOC and Cityscapes. However, its effectiveness on plant datasets hasn't received much importance. To bridge this gap, we empirically study and benchmark the effectiveness of four uncertainty-based active learning strategies on three natural plant organ segmentation datasets. We also study their behaviour in response to variations in training configurations in terms of augmentations used, the scale of training images, active learning batch sizes, and train-validation set splits.

## Conclusion
In this paper, we have studied the efficacy of uncertainty based AL strategies extensively on three plant organ segmentation datasets. Our experimental results show that AL struggles to outperform random sampling in two out of three datasets we studied. We suspect that the traditional uncertainty estimation at image-level is poor by design for binary class imbalanced segmentation datasets. We believe there is a need for AL methods particularly for plant phenotyping tasks that operate at region-level where only important regions of images are sampled and labeled. We leave this for future work. We also studied how AL methods performed under differing training configurations. While we observed small improvements in AL performance with changes in data augmentations, initial pool size, AL batch size and validation set size, we found that performing AL at smaller (even 75% smaller) image scales yields largely similar labeled datasets and ultimately similar model performances compared to operating in original image scales, proving to be a great way of cutting computation and annotation costs. 

## Acknowledgement
This study was partially funded by Indo-Japan DST-JST SICORP program “Data Science-based Farming Support System for Sustainable Crop Production under Climatic Change” and CREST Program “Knowledge Discovery by Constructing AgriBigData” (JPMJCR1512) from Japan Science and Technology Agency.


## References
1. S. Bargoti and J. Underwood, “Image segmentation for fruit detection and yield estimation inapple orchards,”To Appear in Journal of Field Robotics, 2016.
2.  E. David, M. Serouart, D. Smith, S. Madec, K. Velumani, S. Liu, X. Wang, F. P. Espinosa,S. Shafiee, I. Tahir, H. Tsujimoto, S. Nasuda, B. Zheng, N. Kichgessner, H. Aasen, A. Hund,P. Sadhegi-Tehran, K. Nagasawa, G. Ishikawa, S. Dandrifosse, A. Carlier, B. Mercatoris, K.Kuroki, H. Wang, M. Ishii, M. A. Badhon, C. Pozniak, D. LeBauer, M. Lilimo, J. Poland, S.Chapman, B. D. Solan, F. Baret, I. Stavness, and W. Guo, “Global wheat head dataset 2021: More diversity to improve the benchmarking of wheat head localization methods,” 2021.
3. S. V. Desai, V. Balasubramanian, T. Fukatsu, S. Ninomiya, and W. Guo, “Automatic estima-tion of heading date of paddy rice using deep learning,”Plant Methods, vol. 15, 2019.  
