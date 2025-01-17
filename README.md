# IDRT_classification_model

## Dataset Overview
Number of training images: 5591\
Number of validation images: 859\

Train Class Distribution: {'cat07 MasAptMotel': 412, 'cat10 StripMall': 717, 'cat11 L_ShopMall': 156, 'cat17 LowRise': 556, 'cat18 MidRise': 900, 'cat19 HighRise': 629, 'cat20 InstiBldg': 657, 'cat21 MetalBldg': 1021, 'cat22 Canopy': 543}

Validation Class Distribution: {'cat07 MasAptMotel': 67, 'cat10 StripMall': 100, 'cat11 L_ShopMall': 25, 'cat17 LowRise': 78, 'cat18 MidRise': 143, 'cat19 HighRise': 101, 'cat20 InstiBldg': 105, 'cat21 MetalBldg': 166, 'cat22 Canopy': 74}

### Training Class Distribution
![figures/class_distribution_training.png](figures/class_distribution_train.png)
### Validation Class Distribution
![figures/val_class_distribution.png](figures/class_distribution_validation.png)

## Model Training
- Baseline model
  - Feature Extraction
    - Conv1: 3 → 64 channels, kernel=3x3, padding=1, MaxPool2x2
    - Conv2: 64 → 128 channels, kernel=3x3, padding=1, MaxPool2x2
    - Conv3: 128 → 256 channels, kernel=3x3, padding=1, MaxPool2x2
  - Classification:
    - Flattened output (256x28x28) ->  FC1: 512 units -> ReLU -> Dropout()
    - Fully connected layer: num_classes units 
  - Result Table
    Labels| Epoch | Validation Accuracy | Precision | Recall | F1 Score | 
    |---|-------|-------------------|-----------|---------|-----------|
    | 9|5 | 59.37 | 60 | 57 | 58 |
    |6(removing low accuracy classes: cat07, cat17, cat20)|5 |78.33 |80 |73 | 75|
    


