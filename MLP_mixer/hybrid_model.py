import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from timm.layers import DropPath

class Mlp(nn.Module):
    """MLP as used in Vision Transformers, MLP-Mixer and related works"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = drop

# class MixerBlock(nn.Module):
#     def __init__():
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
#         self.drop_path=DropPath(drop_path) if drop_path > 0 else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)
#     def forward(self, x):
#         x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1,2)).transpose(1,2))
#         x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
#         return x
    # self, dim, seq_len, tokens_dim, channels_dim, 
    #              mlp_layer=Mlp, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0, drop_path=0
class MultiChannel_CNNAttentionModel(nn.Module):
    def __init__(self, bert, output_dim, dropout, n_filters, filter_sizes, batch_size, hidden_dim, vocab_size, embedding_length,\
                 mlp_layer=Mlp, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0, drop_path=0):
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
        # MLP mixer
        self.norm1 = norm_layer(embedding_length)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path=DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)
        # CNN
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.conv_0 = nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(1, n_filters, (filter_sizes[2], embedding_dim))

        # Attention
        self.lstm = nn.LSTM(embedding_length, hidden_dim)
        
        # Hybrid
        self.fc1 = nn.Linear(400, 200)
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

        # MLP mixer
        
        
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