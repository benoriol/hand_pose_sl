

import torch
import torch.nn as nn

# import fairseq.models.transformer.TransformerEncoder as TransformerEncoder
#
# class Body2PoseMHSA(nn.Module):
#
#     def __init__(self):
#         super(Body2PoseMHSA, self).__init__()
#
#         self.encoder = TransformerEncoder()


class ConvModel(nn.Module):
    def __init__(self):

        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv1d(12 * 2 ,30, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(30, 30, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(30, 30, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(30, 2 * 21, kernel_size=5, padding=2)

        self.relu = nn.ReLU()


    def forward(self, inp):


        inp = inp.permute(0, 2, 3, 1)
        bs, n_keypoints, dim, len = inp.shape

        inp = inp.view(bs, n_keypoints * dim, len)

        out = self.relu(self.conv1(inp))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)

        out = out.view(bs, -1, dim, len)

        out = out.permute(0, 3, 1, 2)

        return out

