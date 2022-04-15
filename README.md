# EE559_Project

- 2-class problem

- Dataset (# data pts.)
    - training: 184
        - Class 0: 69 (37.5%)
        - Class 1: 115 (62.5%)
    - test: 60

- Sequential Backward Selection
    - Most contributing features:
    - Ws > RH > FFMC > Tempature > DC > BUI > DMC > Rain > ISI

- Required reference systems
    - Trivial system \
        `python3 trivial.py`
        - Test F1-score: 0.5
        - Test Accuracy: 0.5 
    
    - Baseline system \
        `python3 baseline.py`
        - Drop "Date"
        - Test F1-score: 0.6286
        - Test Accuracy: 0.7833

- Technique 1: Perceptron Learning (Drop "Date")\
    `python3 perceptron.py --M 4 --epoch 200 --plot_title perceptron` (M-fold cross-validation)
    - Val F1-score: 0.9113
    - Val Accuracy: 0.9076
    - Test F1-score: 0.8846
    - Test Accuracy: 0.9 

    `python3 perceptron.py --M 4 --epoch 200 --normalization --plot_title p_norm` 
    - Apply min-max normalization to all features
    - Val F1-score: 0.9405
    - Val Accuracy: 0.9457
    - Test F1-score: 0.8679
    - Test Accuracy: 0.8833 

    `python3 perceptron.py --M 4 --epoch 200 --standardization --plot_title p_std` 
    - Apply standardization to all features
    - Val F1-score: 0.9368
    - Val Accuracy: 0.9457
    - Test F1-score: 0.92
    - Test Accuracy: 0.93 

    `python3 perceptron.py --M 4 --epoch 200 --standardization --use_SMOTE --plot_title p_std_SMOTE` 
    - Apply standardization to all features
    - Val F1-score: 0.9667
    - Val Accuracy: 0.9674
    - Test F1-score: 0.9388
    - Test Accuracy: 0.95

- Technique 2: KNN Classifier (Drop "Date", with Standardization)\
    `python3 kNN.py --M 4 --k 5 --plot_title kNN`
    - The following results are for k = (3, 4, 5, 6, 7)
    - Val F1-score: (0.8532, 0.8605, 0.8745, 0.886, 0.8678)
    - Val Accuracy: (0.8641, 0.8696, 0.875, 0.875, 0.8641)
    - Test F1-score: (0.7179, 0.6061, 0.7568, 0.6857, 0.7368)
    - Test Accuracy: (0.8167, 0.7833, 0.85, 0.8167, 0.8333)

    `python3 kNN.py --M 4 --k 5 --use_SMOTE --plot_title kNN_SMOTE`
    - Val F1-score: 0.8594
    - Val Accuracy: 0.8696
    - Test F1-score: 0.7692
    - Test Accuracy: 0.85

    `python3 kNN.py --M 4 --k 5 --feat_reduction --plot_title kNN_feat_reduct`
    - Four least contributing features: ISI -> Rain -> DMC -> BUI
    - Drop (1,2,3,4) features
    - Val F1-score: (0.7121, 0.7098, 0.6595, 0.7183)
    - Val Accuracy: (0.7663, 0.7663, 0.75, 0.7609)
    - Test F1-score: (0.6666, 0.6154, 0.5714, 0.6222)
    - Test Accuracy: (0.7666, 0.75, 0.7, 0.7167)

