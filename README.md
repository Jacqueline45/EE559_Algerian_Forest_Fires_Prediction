# EE559_Project

- 2-class problem

- Dataset (# data pts.)
    - training: 184
        - Class 0: 69 (37.5%)
        - Class 1: 115 (62.5%)
    - test: 60

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

- Technique 1: Perceptron Learning \
    `python3 perceptron.py --M 4 --epoch 200 --plot_title perceptron` (M-fold cross-validation)
    - Drop "Date"
    - Val F1-score: 0.9113
    - Val Accuracy: 0.9076
    - Test F1-score: 0.8846
    - Test Accuracy: 0.9 \

    `python3 perceptron.py --M 4 --epoch 200 --normalization --plot_title p_norm` 
    - Drop "Date"
    - Apply min-max normalization to all features
    - Val F1-score: 0.9405
    - Val Accuracy: 0.9457
    - Test F1-score: 0.8679
    - Test Accuracy: 0.8833 \

    `python3 perceptron.py --M 4 --epoch 200 --standardization --plot_title p_std` 
    - Drop "Date"
    - Apply standardization to all features
    - Val F1-score: 0.9368
    - Val Accuracy: 0.9457
    - Test F1-score: 0.92
    - Test Accuracy: 0.93 \

