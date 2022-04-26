# EE559_Project
- 2-class problem

- Dataset (# data pts.)
    - training: 184
        - Class 0: 69 (37.5%)
        - Class 1: 115 (62.5%)
    - test: 60

- Sequential Backward Selection
    - Most contributing features:
    - ISI > Rain > DMC > FFMC > DC > RH > BUI > Ws > Temperature

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
    - Val F1-score: 0.9372
    - Val Accuracy: 0.9402
    - Test F1-score: 0.9583
    - Test Accuracy: 0.9667

    `python3 perceptron.py --M 4 --epoch 200 --standardization --use_SMOTE --feat_reduction --plot_title p_feat_reduct` 
    - Four least contributing features: Temperature -> Ws -> BUI -> RH
    - Drop (1,2,3,4) features
    - Val F1-score: (0.9725, 0.9805, 0.9763, 0.9875)
    - Val Accuracy: (0.9674, 0.9728, 0.9728, 0.9837)
    - Test F1-score: (0.9583, 0.9787, 0.9388, 0.9583)
    - Test Accuracy: (0.9667, 0.9833, 0.95, 0.9667)

    `python3 perceptron.py --standardization --use_SMOTE --feat_reduction --extra_feat --plot_title p_add_1_feat` 
    - Val F1-score: 0.9882
    - Val Accuracy: 0.9783
    - Test F1-score: 0.9787
    - Test Accuracy: 0.9833

    `python3 perceptron.py --standardization --use_SMOTE --extra_feat --plot_title p_add_1_feat` 
    - Val F1-score: 0.9767
    - Val Accuracy: 0.9565
    - Test F1-score: 0.92
    - Test Accuracy: 0.9333

- Technique 2: KNN Classifier (Drop "Date", with Standardization)\
    `python3 kNN.py --M 4 --k 7 --plot_title kNN`
    - The following results are for k = (2, 3, 4, 5, 6, 7, 8)
    - Val F1-score: (0.8287, 0.8532, 0.8605, 0.8745, 0.886, 0.8678, 0.8739)
    - Val Accuracy: (0.8478, 0.8641, 0.8696, 0.875, 0.875, 0.8641, 0.8804)
    - Test F1-score: (0.7, 0.8444, 0.8095, 0.8182, 0.8372, 0.8444, 0.8182)
    - Test Accuracy: (0.8, 0.8833, 0.8667, 0.8667, 0.8833, 0.8833, 0.8667)

    `python3 kNN.py --M 4 --k 7 --use_SMOTE --plot_title kNN_SMOTE`
    - Val F1-score: 0.8448
    - Val Accuracy: 0.8641
    - Test F1-score: 0.8372
    - Test Accuracy: 0.8833

    `python3 kNN.py --M 4 --k 7 --feat_reduction --plot_title kNN_feat_reduct`
    - Four least contributing features: Temperature -> Ws -> BUI -> RH
    - Drop (1,2,3,4) features
    - Val F1-score: (0.8898, 0.9108, 0.9236, 0.9313)
    - Val Accuracy: (0.8967, 0.9076, 0.9293, 0.9239)
    - Test F1-score: (0.8444, 0.8182, 0.8444, 0.8182)
    - Test Accuracy: (0.8833, 0.8667, 0.8833, 0.8667)

    `python3 kNN.py --extra_feat --feat_reduction --plot_title kNN_add_1_feat` 
    - Drop temperature
    - Val F1-score: 0.9767
    - Val Accuracy: 0.9565
    - Test F1-score: 0.8085
    - Test Accuracy: 0.85
    
    `python3 kNN.py --extra_feat --plot_title kNN_add_1_feat` 
    - Val F1-score: 0.9767
    - Val Accuracy: 0.9565
    - Test F1-score: 0.8511
    - Test Accuracy: 0.8833

- Technique 3: MSE Classifier (Drop "Date")\
    `python3 MSE.py --use_SMOTE --plot_title p_MSE_SMOTE` (b=1 for all data pts)
    - Val F1-score: 0.9898
    - Val Accuracy: 0.9837
    - Test F1-score: 0.9787
    - Test Accuracy: 0.9833

    `python3 MSE.py --plot_title p_MSE` (b=1 for all data pts)
    - Val F1-score: 0.9942
    - Val Accuracy: 0.9891
    - Test F1-score: 1
    - Test Accuracy: 1
