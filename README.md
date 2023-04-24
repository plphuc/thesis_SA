orgin/: orgin model (CNN, Self-Attention) in BAKSA <br />
output/: stdout when running models <br />
CNNModel.py, SelfAttentionModel.py: models are used in hybrid model (our model) <br />
Multichane...py (1st model): 1 Linear (400 - 3), softmax, dropout 0.3 <br />
hybrid_{n}: our model with others architectures/parameters <br />
- n = 1: 1st model without softmax
- n = 2: 2 Linear layers (400-200, 200-3), dropout 0.3, softmax <br />
main_3_{n}.py: main.py to run the corresponding n model

[model description](https://www.notion.so/plphuc/Survey-results-5e55631af06d4b86a4e46854680f80ad)
 
