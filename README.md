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

- Technique 1: Perceptron
        `python3 perceptron.py --M 4 --epoch 200` (M-fold cross-validation)
        - Drop "Date"
        - Val F1-score: 0.924
        - Val Accuracy: 0.9239
        - Test F1-score: 0.902
        - Test Accuracy: 0.9167