orgin/: orgin model (CNN, Self-Attention) in BAKSA
output/: stdout when running models
CNNModel.py, SelfAttentionModel.py: models are used in hybrid model (our model)
Multichane...py (1st model): 1 Linear (400 - 3), softmax, dropout 0.3
hybrid_{n}: our model with others architectures/parameters
- n = 1: 1st model without softmax
- n = 2: 2 Linear layers (400-200, 200-3), dropout 0.3, softmax
main_3_{n}.py: main.py to run the corresponding n model

 
