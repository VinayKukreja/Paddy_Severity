Paddy Leaf Disease Severity Classification
================================================================
Pipeline (paper section references in comments throughout):
  1. Dataset collection: primary + secondary (Mendeley, GitHub, UCI/Kaggle) -> Sec 3.1
  2. Pre-processing: standardization, normalization, rescaling             -> Sec 3.1.3
  3. GAN-based data augmentation (per-class DCGAN)                        -> Sec 3.2
  4. Rule-based severity via image segmentation (label only, not learned) -> Sec 3.3
  5. CNN feature extractor + SVM-style head, disease type (3 classes)     -> Sec 3.4, Table 4
  6. Train/val/test (80:20, then 80:20), confusion matrix, precision/recall/F1 -> Sec 3.5, Sec 4

Architecture notes:
  - Layer sizes/parameter counts match Table 4 exactly (verified by hand):
    conv2d_12 (7,296), conv2d_13 (55,360), dense_17 (1,179,936), dense_18 (867).
  - Table 4 lists conv2d_13's stride as 2, but its stated output shape
    (16x16x64) and the downstream 4,096-unit flatten are only consistent
    with stride=1 there; stride=1 is used to match the stated shapes.
  - Output layer uses linear activation + L2 regularization + squared_hinge
    loss (the standard "CNN-SVM" / L2-SVM formulation), since the paper's
    description of "softmax" together with "squared hinge loss" is
    internally contradictory.
  - GAN augmentation uses the paper's literal per-class percentages
    (Sec 3.2: blight +424%, blast +180%, leaf smut +922%), and train/val/
    test split is performed AFTER augmentation, matching the paper's
    described order.
  - Severity is computed via rule-based segmentation only (Sec 3.3) and is
    not a model output — Table 4's final layer has 3 units (disease
    classes only), and no severity confusion matrix is reported anywhere
    in the paper.

Requirements:
    pip install tensorflow opencv-python scikit-learn pandas matplotlib tqdm

