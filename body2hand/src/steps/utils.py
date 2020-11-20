import pickle
from fairseq import utils
import torch
import numpy as np

from torch.utils.data.dataloader import default_collate

import torch.nn as nn


class AverageMeter():
    def __init__(self):
        self.total = 0
        self.count = 0

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def get_average(self):
        return  self.total / self.count

class ProgressSaver():

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.progress = {
            "epoch":[],
            "train_loss":[],
            "train_pix_dist":[],
            "val_loss":[],
            "val_pix_dist": [],
            "time":[],
            "best_epoch":[],
            "best_val_loss":[],
            "lr": []
            }

    def update_epoch_progess(self, epoch_data):

        for key in epoch_data.keys():
            self.progress[key].append(epoch_data[key])

        # self.progress["epoch"].append(epoch_data["epoch"])
        # self.progress["train_loss"].append(epoch_data["train_loss"])
        # self.progress["val_loss"].append(epoch_data["val_loss"])
        # self.progress["train_pix_dist"].append(epoch_data["train_pix_dist"])
        # self.progress["val_pix_dist"].append(epoch_data["val_pix_dist"])
        # self.progress["time"].append(epoch_data["time"])
        # self.progress["best_epoch"].append(epoch_data["best_epoch"])
        # self.progress["best_val_loss"].append(epoch_data["best_val_loss"])

        with open("%s/progress.pckl" % self.exp_dir, "wb") as f:
            pickle.dump(self.progress, f)

    def load_progress(self):
        with open("%s/progress.pckl" % self.exp_dir, "rb") as f:
            self.progress = pickle.load(f)

    def get_resume_stats(self):
        return self.progress["epoch"][-1], self.progress["best_val_loss"][-1], self.progress["time"][-1]



def print_epoch(epoch, train_loss, train_pix_dist, val_loss, val_pix_dist, time):

    print("Epoch #" + str(epoch) + ":\tTrain loss: " + str(train_loss) +"\tValid loss: " + str(val_loss) +
          "\n\t Train pix dist: " + str(train_pix_dist) + "\tValid pix dist: " + str(val_pix_dist) +
          "\n\tTime: " + str(time))

def add_transformer_args(parser):
    """Add model-specific arguments to the parser."""
    # fmt: off
    parser.add_argument('--activation-fn',
                        choices=utils.get_available_activation_fns(),
                        help='activation function to use',
                        default="relu")
    parser.add_argument('--dropout', type=float, metavar='D',
                        help='dropout probability', default=0.0)
    parser.add_argument('--attention-dropout', type=float, metavar='D',
                        help='dropout probability for attention weights',
                        default=0.0)
    parser.add_argument('--activation-dropout', '--relu-dropout',
                        type=float,
                        metavar='D',
                        help='dropout probability after activation in FFN.',
                        default=0.0)
    parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained encoder embedding')
    parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                        help='encoder embedding dimension', default=21*2)
    parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                        help='encoder embedding dimension for FFN',
                        default=100)
    parser.add_argument('--encoder-layers', type=int, metavar='N',
                        help='num encoder layers', default=4)
    parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                        help='num encoder attention heads', default=3)
    parser.add_argument('--encoder-normalize-before', action='store_true',
                        help='apply layernorm before each encoder block')
    parser.add_argument('--encoder-learned-pos', action='store_true',
                        help='use learned positional embeddings in the encoder')
    parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained decoder embedding')
    parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension', default=100)
    parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension for FFN')
    parser.add_argument('--decoder-layers', type=int, metavar='N',
                        help='num decoder layers')
    parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                        help='num decoder attention heads')
    parser.add_argument('--decoder-learned-pos', action='store_true',
                        help='use learned positional embeddings in the decoder')
    parser.add_argument('--decoder-normalize-before', action='store_true',
                        help='apply layernorm before each decoder block')
    parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                        help='decoder output dimension (extra linear layer '
                             'if different from decoder embed dim')
    parser.add_argument('--share-decoder-input-output-embed',
                        action='store_true',
                        help='share decoder input and output embeddings')
    parser.add_argument('--share-all-embeddings', action='store_true',
                        help='share encoder, decoder and output embeddings'
                             ' (requires shared dictionary and embed dim)')
    parser.add_argument('--no-token-positional-embeddings', default=False,
                        action='store_true',
                        help='if set, disables positional embeddings (outside self attention)')
    parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                        help='comma separated list of adaptive softmax cutoff points. '
                             'Must be used with adaptive_loss criterion'),
    parser.add_argument('--adaptive-softmax-dropout', type=float,
                        metavar='D',
                        help='sets adaptive softmax dropout for the tail projections')
    parser.add_argument('--layernorm-embedding', action='store_true',
                        help='add layernorm to embedding')
    parser.add_argument('--no-scale-embedding', action='store_true',
                        help='if True, dont scale embeddings')
    # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    parser.add_argument('--no-cross-attention', default=False,
                        action='store_true',
                        help='do not perform cross-attention')
    parser.add_argument('--cross-self-attention', default=False,
                        action='store_true',
                        help='perform cross+self-attention')
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    parser.add_argument('--encoder-layerdrop', type=float, metavar='D',
                        default=0,
                        help='LayerDrop probability for encoder')
    parser.add_argument('--decoder-layerdrop', type=float, metavar='D',
                        default=0,
                        help='LayerDrop probability for decoder')
    parser.add_argument('--encoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')
    parser.add_argument('--decoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    parser.add_argument('--quant-noise-pq', type=float, metavar='D',
                        default=0,
                        help='iterative PQ quantization noise at training time')
    parser.add_argument('--quant-noise-pq-block-size', type=int,
                        metavar='D',
                        default=8,
                        help='block size of quantization noise at training time')
    parser.add_argument('--quant-noise-scalar', type=float, metavar='D',
                        default=0,
                        help='scalar quantization noise and scalar quantization at training time')

    # Additional
    parser.add_argument('--max-source-positions', type=int,
                        default=150)
    parser.add_argument('--adaptive-input', type=bool,
                        default=False)


class NormalizeFixedFactor:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, item):

        item["body_kp"] = item["body_kp"] / self.factor
        item["right_hand_kp"] = item["right_hand_kp"] / self.factor
        item["left_hand_kp"] = item["left_hand_kp"] / self.factor

        return item
    def denormalize_tensor(self, tensor):
        return tensor * self.factor

class WristDifference:
    def __init__(self):
        # According to  https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-body_25
        # and taking into consideration we are not using all the body points. See text_pose_dataset.py
        self.right_wrist_index = 4
    def __call__(self, item):
        item["right_hand_kp"] = item["right_hand_kp"] - item["body_kp"][:, self.right_wrist_index].unsqueeze(1)
        return item

class ChestDifference:
    def __init__(self):
        # According to  https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-body_25
        # and taking into consideration we are not using all the body points. See text_pose_dataset.py
        self.chest_index = 1
    def __call__(self, item):
        item["body_kp"] = item["body_kp"] - item["body_kp"][:, self.chest_index].unsqueeze(1)
        return item



# modify item to train to learn to predict the right hand index from the body
# and other right hand keypoints.

class BuildIndexItem:
    def __call__(self, item):
        right_hand_kp = item["right_hand_kp"]
        right_hand_noindex_kp = [right_hand_kp[:, 0:5], right_hand_kp[:, 9:]]
        right_hand_noindex_kp = torch.cat(right_hand_noindex_kp, dim=1)

        right_hand_index_kp = right_hand_kp[:, 5:9]

        body_righthandnoindex_kp = torch.cat([item["body_kp"], right_hand_noindex_kp], dim=1)

        del item["right_hand_kp"]
        del item["body_kp"]

        item["right_hand_index_kp"] = right_hand_index_kp
        item["body_righthandnoindex_kp"] = body_righthandnoindex_kp

        item["input_kp"] = item["body_righthandnoindex_kp"]
        item["target_kp"] = item["right_hand_index_kp"]

        return item

class Build3fingerItem:

    def __call__(self, item):
        right_hand_kp = item["right_hand_kp"]


        right_hand_no3finger_kp = torch.cat([right_hand_kp[:, 13:], right_hand_kp[:, 0].unsqueeze(1)],
                                             dim=1)
        right_hand_3finger_kp = right_hand_kp[:, 1:13]

        right_bodyhand_no3finger_kp = torch.cat([right_hand_no3finger_kp, item["body_kp"]], dim=1)

        del item["right_hand_kp"]
        del item["body_kp"]

        item["right_hand_3finger_kp"] = right_hand_3finger_kp
        item["right_bodyhand_no3finger_kp"] = right_bodyhand_no3finger_kp

        item["input_kp"] = item["right_bodyhand_no3finger_kp"]
        item["target_kp"] = item["right_hand_3finger_kp"]

        return item

class BuildRightHandItem:

    def __call__(self, item):

        item["input_kp"] = item["body_kp"]
        item["target_kp"] = item["right_hand_kp"]

        item["input_conf"] = item["body_conf"]
        item["target_conf"] = item["right_hand_conf"]

        # del item["right_hand_kp"]
        # del item["body_kp"]

        # del item["body_conf"]
        # del item["right_hand_conf"]

        return item

# TODO Fix pixel distance formula
class MSE2Pixels:
    def __init__(self, num_joints, upsample):
        self.num_joints = num_joints
        self.upsample = upsample

    def __call__(self, mse):
        pix = mse / self.num_joints
        pix = pix ** (1/2)
        pix = pix * self.upsample
        return pix

class L12Pixels:
    def __init__(self, num_joints, upsample):
        self.num_joints = num_joints
        self.upsample = upsample

    def __call__(self, mse):
        pix = mse / self.num_joints
        pix = pix * self.upsample
        return pix

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch / lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def mask_output(output, lengths):
    for i, len in enumerate(lengths):
        output[i, len:, :] = 0
    return output

def tensor2json(right_hand, left_hand, body):
    data_dict = {
        "version": 1.3,
        "people": []
    }

    person = {
        "person_id": [-1],


    }
    right_hand = np.concatenate((right_hand, np.zeros((21, 1)) + 1.0), axis=1)
    right_hand = np.reshape(right_hand, (-1))
    right_hand = [float(x) for x in right_hand]

    left_hand = np.concatenate((left_hand, np.zeros((21, 1)) + 1.0), axis=1)
    left_hand = np.reshape(left_hand, (-1))
    left_hand = [float(x) for x in left_hand]

    body = np.concatenate((body, np.zeros((12, 1)) + 1.0), axis=1)
    body = np.reshape(body, (-1))
    body = [float(x) for x in body]

    person["pose_keypoints_2d"] = body
    person["hand_left_keypoints_2d"] = left_hand
    person["hand_right_keypoints_2d"] = right_hand

def array2open_pose_index(openpose_right_hand, predicted_right_index, confidence=None):

    if confidence==None:
        confidence = np.zeros((4, 1)) + 1.0

    predicted_right_index = np.concatenate((predicted_right_index, confidence), axis=1)
    predicted_right_index = np.reshape(predicted_right_index, (-1))
    predicted_right_index = [float(x) for x in predicted_right_index]
    assert len(predicted_right_index) == 4 * 3
    # Replace the right index GT with the predictions
    openpose_right_hand[5*3:9*3] = predicted_right_index

    return openpose_right_hand

def array2open_pose(array, confidence=None):

    if confidence==None:
        confidence = np.zeros((21, 1)) + 1.0

    array = np.concatenate((array, confidence), axis=1)
    array = np.reshape(array, (-1))
    array = [float(x) for x in array]

    return array

def array2open_pose_3finger(openpose_right_hand, predicted_right_3finger, confidence=None):

    if confidence==None:
        confidence = np.zeros((12, 1)) + 1.0

    predicted_right_3finger = np.concatenate((predicted_right_3finger, confidence), axis=1)
    predicted_right_3finger = np.reshape(predicted_right_3finger, (-1))
    predicted_right_3finger = [float(x) for x in predicted_right_3finger]
    assert len(predicted_right_3finger) == 12 * 3
    # Replace the right index GT with the predictions
    # From 1st joint to 13th joint not included. see
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#hand-output-format
    openpose_right_hand[1*3:13*3] = predicted_right_3finger

    return openpose_right_hand


def collate_function(batch):

    json_paths = [elem["json_paths"] for elem in batch]
    texts = [elem["text"] for elem in batch]

    for elem in  batch:
        del elem["json_paths"]
        del elem["text"]

    batch = default_collate(batch)
    batch["json_paths"] = json_paths
    batch["text"] = texts

    return batch

def collate_function_h5(batch):

    utt_ids = [elem["utt_id"] for elem in batch]
    texts = [elem["text"] for elem in batch]

    for elem in  batch:
        del elem["utt_id"]
        del elem["text"]

    batch = default_collate(batch)
    batch["utt_id"] = utt_ids
    batch["text"] = texts

    return batch

class maskedPoseL1(nn.Module):
    def __init__(self):
        super(maskedPoseL1, self).__init__()
        # This reduction will average the loss across the seq_lenght dimension
        # (1st dimension)
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, prediction, target, lengths):
        loss = 0
        for i, seq_len in enumerate(lengths):
            prediction_ = prediction[i, :seq_len]
            target_ = target[i, :seq_len]

            # Computes the average loss across the length of the sequence
            loss += self.l1(prediction_, target_)
        return loss / (i+1)


class poderatedPoseL1(nn.Module):
    def __init__(self):
        super(poderatedPoseL1, self).__init__()

        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, prediction, target, lengths, scores):

        scores = scores.to(prediction.device)

        loss = 0
        for i, seq_len in enumerate(lengths):
            utt_prediction = prediction[i, :seq_len]
            utt_scores = scores[i, :seq_len]
            utt_target = target[i, :seq_len]

            pond_predictions = utt_prediction * utt_scores.unsqueeze(2)
            pond_target = utt_target * utt_scores.unsqueeze(2)

            loss += self.l1(pond_predictions, pond_target)

        return loss

from torch import t