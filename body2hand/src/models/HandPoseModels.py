import math

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding, #Implemented my own
    SinusoidalPositionalEmbedding,
    TransformerEncoderLayer,
)

class ConvModel(nn.Module):
    def __init__(self, conv_channels, activation, pos_emb):

        super(ConvModel, self).__init__()
        if pos_emb:
            #self.pos_emb = PositionalEncoding(2*12, dropout=0.0, max_len=100)
            self.pos_emb = LinearPositionalEmbedding(max_len=100)
            self.conv1 = nn.Conv1d(12 * 2 + 1, conv_channels, kernel_size=5, padding=2)
            #self.conv1 = nn.Conv1d(12 * 2, conv_channels, kernel_size=5, padding=2)
        else:
            self.pos_emb = None
            self.conv1 = nn.Conv1d(12 * 2, conv_channels, kernel_size=5, padding=2)

        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(conv_channels, 2 * 21, kernel_size=5, padding=2)

        if activation == "ReLU":
            self.activation = nn.ReLU()
        else:
            raise ValueError()


    def forward(self, inp):


        inp = inp.permute(0, 2, 3, 1)
        bs, n_keypoints, dim, len = inp.shape

        inp = inp.view(bs, n_keypoints * dim, len)

        if self.pos_emb:
            # inp = inp.permute(0, 2, 1)
            # inp = self.pos_emb(inp)
            # inp = inp.permute(0, 2, 1)

            inp = self.pos_emb(inp, None)

        out = self.activation(self.conv1(inp))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.conv4(out)

        out = out.view(bs, -1, dim, len)

        out = out.permute(0, 3, 1, 2)

        return out

class LinearPositionalEmbedding(nn.Module):

    def __init__(self, max_len=100):
        super(LinearPositionalEmbedding, self).__init__()

        self.pe = range(max_len)
        self.pe = torch.tensor(self.pe)
        self.pe = torch.unsqueeze(self.pe, dim=0)
        self.pe = torch.unsqueeze(self.pe, dim=0)
        self.pe = self.pe.float() / max_len


    def forward(self, inp, lengths):
        bs = inp.shape[0]
        pe = torch.cat(bs * [self.pe], dim=0).to(inp.device)

        out = torch.cat([pe, inp], dim=1)

        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SinusoidalPositionalEmbedding(nn.Module):
    def forward(self, inp, lenghts):
        batch_size, seq_len = inp.shape[0], inp.shape[1]
        positional_embeddings = torch.zeros(batch_size, seq_len)

        for i, len in enumerate(lenghts):
            for x in range(len):
                positional_embeddings[i, x] = math.cos(x / len * math.pi)

        inp.cat(positional_embeddings, dim=2)

        return inp

class TransformerEnc(nn.Module):

    def __init__(self, ninp, nhead, nhid, nout, nlayers, dropout=0.5):

        super(TransformerEnc, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=100)
        encoder_layers = TransformerEncoderLayer(nhid, nhead, nhid,
                                                 dropout)

        #self.linear_pos_enc = LinearPositionalEmbedding(max_len=100)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp

        self.hidden2pose_projection = nn.Linear(nhid, nout)
        self.pose2hidden_projection = nn.Linear(ninp, nhid)

        #self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0,
                                        float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        bs, seq_len, n_in_joints, dim = src.shape
        src = src.view(bs, seq_len, n_in_joints * dim)

        #src = self.linear_pos_enc(src)

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(
                device)
            self.src_mask = mask

        # src = self.encoder(src) * math.sqrt(self.ninp)
        # B x T x C -> T x B x C
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)

        src = self.pose2hidden_projection(src)
        output = self.transformer_encoder(src)
        output = self.hidden2pose_projection(output)
        output = output.permute(1, 0, 2)

        output = output.view(bs, seq_len, -1, dim)

        return output


class TextPoseTransformer(nn.Module):
    def __init__(self, n_tokens, n_joints, joints_dim, nhead, nhid, nout, n_enc_layers, n_dec_layers, dropout=0.5):
        super(TextPoseTransformer, self).__init__()
        from torch.nn import Transformer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.token_pos_encoder = PositionalEncoding(nhid, dropout,
                                              max_len=40)
        self.pose_pos_encoder = PositionalEncoding(nhid, dropout,
                                              max_len=100)

        self.transformer = Transformer(nhid, nhead, n_enc_layers, n_dec_layers, nhid, dropout=dropout)

        self.token_embedding = nn.Embedding(n_tokens, nhid)
        self.hidden2pose_projection = nn.Linear(nhid, nout)
        self.pose2hidden_projection = nn.Linear(n_joints*joints_dim, nhid)

        self.init_weights()


    def forward(self, input_tokens, input_pose):
        bs, seq_len, n_in_joints, dim = input_pose.shape
        input_pose = input_pose.view(bs, seq_len, -1)
        input_pose = input_pose.permute(1, 0, 2)

        input_tokens_embedding = self.token_embedding(input_tokens)
        input_tokens_embedding = input_tokens_embedding.permute(1, 0, 2)
        #input_pose = input_pose.reshape(bs * seq_len, -1)
        input_pose = self.pose2hidden_projection(input_pose)

        predictions = self.transformer(input_tokens_embedding, input_pose)

        predictions = self.hidden2pose_projection(predictions)

        predictions = predictions.permute(1, 0, 2)
        predictions = predictions.view(bs, seq_len, -1, dim)





        return predictions


    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.transformer.bias.data.zero_()
        # self.transformer.weight.data.uniform_(-initrange, initrange)
        pass
