import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductScore(nn.Module):
    """
    Vaswani et al. "Attention Is All You Need", 2017.
    """
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys):
        """
        queries:  [batch_size x num_queries x dim]
        keys:     [batch_size x num_objects x dim]
        Returns a tensor of scores with shape [batch_size x num_queries x num_objects].
        """
        result = torch.bmm(queries, torch.transpose(keys, 1, 2))/np.sqrt(keys.shape[2])
        return result


class Attention(nn.Module):
    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, queries, keys, values):
        """
        queries:         [batch_size x num_queries x query_feature_dim]
        keys:            [batch_size x num_objects x key_feature_dim]
        values:          [batch_size x num_objects x obj_feature_dim]
        Returns matrix of responses for queries with shape [batch_size x num_queries x obj_feature_dim].
        Saves detached weights as self.attention_map.
        """
        scores = self.scorer(queries, keys)

        weights = F.softmax(scores, 2)
        self.attention_map = weights.detach()
        result = torch.bmm(weights, values)
        return result

#@title Default title text
from torch.nn.modules.rnn import LSTM
class CaptionNet(nn.Module):
    def __init__(self, n_tokens, cnn_w_h = 9*9,  cnn_channels = 512, emb_size = 128, lstm_units = 256, logit_hidden_sizes = [256], loggits_act = nn.ReLU(), device = 'CUDA'):
        """ A recurrent 'head' network for image captioning. Read scheme below. """
        super(self.__class__, self).__init__()

        self.device = device
        self.cnn_w_h = cnn_w_h
        self.cnn_channels = cnn_channels
        self.emb_size = emb_size
        self.lstm_units = lstm_units
        self.n_tokens = n_tokens
        # a layer that converts conv features to
        self.cnn_to_h0 = nn.Linear(cnn_channels, lstm_units)
        self.cnn_to_c0 = nn.Linear(cnn_channels, lstm_units)

        # recurrent part, please create the layers as per scheme above.

        # create embedding for input words. Use the parameters (e.g. emb_size).
        self.emb = nn.Embedding(n_tokens, emb_size)

        # attention: create attention over image spatial positions
        # The query is previous lstm hidden state, the keys are transformed cnn features,
        # the values are cnn features
        self.attention =  Attention(ScaledDotProductScore())

        # attention: create transform from cnn features to the keys
        # Hint: one linear layer should work
        # Hint: the dimensionality of keys should be lstm_units as lstm
        #       hidden state is the attention query
        self.cnn_to_attn_key = nn.Linear(cnn_channels, lstm_units)

        # lstm: create a recurrent core of your network. Use LSTMCell
        self.lstm = nn.LSTMCell(input_size = emb_size + cnn_channels, hidden_size = lstm_units)

        # create logits: MLP that takes attention response, lstm hidden state
        # and the previous word embedding as an input and computes one number per token
        # Hint: I used an architecture with one hidden layer, but you may try deeper ones
        self.logits_mlp = nn.Sequential( )
        prev_size = emb_size + lstm_units  + cnn_channels
        for i, size in enumerate(logit_hidden_sizes):
            self.logits_mlp.add_module('logit layer{}'.format(i),
                                  torch.nn.Linear(prev_size, size))
            self.logits_mlp.add_module('relu{}'.format(i), loggits_act)
            prev_size = size
        self.logits_mlp.add_module('final_layer',
                                  torch.nn.Linear(prev_size, n_tokens))

    def forward(self, image_features, captions_ix):
        """
        Apply the network in training mode.
        :param image_features: torch tensor containing VGG features for each position.
                               shape: [batch, cnn_channels, width * height]
        :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i].
            padded with pad_ix
        :returns: logits for next token at each tick, shape: [batch, word_i, n_tokens]
        """
        batch = image_features.shape[0]
        caption_length = captions_ix.shape[1]

        initial_cell = self.cnn_to_c0(image_features.mean(2))
        initial_hid = self.cnn_to_h0(image_features.mean(2))

        image_features = image_features.transpose(1, 2)

     
        attention_map_s = torch.zeros((batch, caption_length, self.cnn_w_h)).to(self.device)
        reccurent_out_s = torch.zeros((batch, caption_length, self.n_tokens)).to(self.device)
        h, c = initial_hid, initial_cell

        for i in range(caption_length):
            caption_emb = self.emb(captions_ix[:, i])
            keys = self.cnn_to_attn_key(image_features)
            h_extra_dim = h[:, None, :]
            context = self.attention(h_extra_dim, keys, image_features)
            context  = context.view(context.shape[0], context.shape[2])
            attention_map = self.attention.attention_map
            attention_map_s[:, i, :] = attention_map.view(attention_map.shape[0], attention_map.shape[2])

            h,c = self.lstm(torch.cat((caption_emb, context), 1),  (h, c))
            reccurent_out = torch.cat((h, context, caption_emb), 1)
            logits = self.logits_mlp(reccurent_out)
            reccurent_out_s[:, i, :] = logits

      
        return reccurent_out_s, attention_map_s