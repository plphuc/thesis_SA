import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MultiChannel_CNNAttentionModel(nn.Module):
    def __init__(self, bert, output_dim, dropout, n_filters, filter_sizes, batch_size, hidden_dim, vocab_size, embedding_length):
        super().__init__()
        self.bert = bert
        self.output_dim = output_dim
        self.dropout = dropout
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.batch_size = batch_size
        self.hidden_size = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.do = nn.Dropout(dropout)

        # CNN
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.conv_0 = nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(1, n_filters, (filter_sizes[2], embedding_dim))

        # Attention
        self.lstm = nn.LSTM(embedding_length, hidden_dim)
        
        # Hybrid
        self.fc1 = nn.Linear(250, 200)
        self.fc2 = nn.Linear(200, output_dim)
        self.softmax = nn.Softmax(dim=1)


    def attention_net(self, lstm_output, final_state):
    
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        
        return new_hidden_state

    def forward(self, text, batch_size):
        with torch.no_grad():
            input = self.bert(text)[0]

        # CNN
        embedded = input.unsqueeze(1)

        #embedded = [batch size, 1, sent len, emb dim]
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        #pooled_n = [batch size, n_filters]
        cnn_output = self.do(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
        
        # Attention
        embedded = input.permute(1,0,2)
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        attn_output, (final_hidden_state, final_cell_state) = self.lstm(embedded, (h_0, c_0))
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = self.attention_net(attn_output, final_hidden_state)

        cnn_attention_cat = torch.cat((cnn_output, attn_output), 1)
        output_ln1 = self.fc1(cnn_attention_cat)
        output_do = self.do(output_ln1)
        output_ln2 = self.fc2(output_do)

        return output_ln2