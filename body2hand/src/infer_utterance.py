
import argparse
from dataloaders import FastPoseDataset, FastTextPoseDataset

import torch
from torch.utils.data import DataLoader

from models import ConvModel, TransformerEncoder, ConvTransformerEncoder, TransformerEnc, TextPoseTransformer
from steps import train, infer_utterance
from steps import NormalizeFixedFactor, add_transformer_args, collate_function
import os
import pickle

import glob

parser = argparse.ArgumentParser()


parser.add_argument("--utterance-folder", type=str)
parser.add_argument("--data", type=str)
parser.add_argument("--output-folder", type=str)
parser.add_argument("--max-frames", type=int, default=200)
parser.add_argument("--model", type=str, choices=["Conv", "TransformerEncoder",
                                                  "ConvTransformerEncoder",
                                                  "TransformerEnc",
                                                  "TextPoseTransformer"],
                    default="TextPoseTransformer")
parser.add_argument("--model-checkpoint", type=str)
parser.add_argument("--conv-channels", type=int, default=30)
parser.add_argument("--conv-pos-emb", action='store_true', default=False)

parser.add_argument("--no-normalize", dest="normalize", action='store_false', default=True)

parser.add_argument("--rand-tokens", dest="rand_tokens", action='store_true', default=False)
parser.add_argument("--transformer-dropout", default=0.1, type=float)


#add_transformer_args(parser)


def main():
    args = parser.parse_args()

    if os.path.isdir(args.output_folder):
        raise Exception("Experiment name " + args.output_folder + " already exists.")
    os.mkdir(args.output_folder)

    with open(args.output_folder + "/args.pckl", "wb") as f:
        pickle.dump(args, f)

    transform = None
    if args.normalize:
        transform = NormalizeFixedFactor(1280)
    # utterance_dict = build_dataset_structure(args.utterance_folder)
    # metadata_structure = [utterance_dict]

    if "Text" in args.model:
        dataset = FastTextPoseDataset(args.data, args.max_frames, transform, use_rand_tokens=args.rand_tokens)
    else:
        dataset = FastPoseDataset(args.data, args.max_frames, transform)

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_function)

    if args.model == "Conv":
        model = ConvModel(args.conv_channels, activation="ReLU", pos_emb=args.conv_pos_emb)
    elif args.model == "TransformerEncoder":
        model = TransformerEncoder(args, 100)
    elif args.model == "ConvTransformerEncoder":
        model = ConvTransformerEncoder(args, 21 * 2)
    elif args.model == "TransformerEnc":
        model = TransformerEnc(ninp=12*2, nhead=4, nhid=100, nout=21*2,
                               nlayers=4, dropout=0.0)
    elif args.model == "TextPoseTransformer":
        model = TextPoseTransformer(n_tokens=1000, n_joints=12, joints_dim=2, nhead=4,
                                    nhid=128, nout=21*2, n_enc_layers=4, n_dec_layers=4,
                                    dropout=args.transformer_dropout)
    else:
        raise ValueError()
    model.load_state_dict(torch.load(args.model_checkpoint))

    infer_utterance(model, loader, args)

def build_dataset_structure(utterance_folder):
    utterance_name = utterance_folder.split("/")[-1]

    frame_jsons = glob.glob(utterance_folder + "/*")
    frame_jsons.sort()

    utterance_dict = dict()

    utterance_dict["utt_id"] = utterance_name

    utterance_dict["text"] = None

    utterance_dict["n_frames"] = len(frame_jsons)
    # utterance_dict["frame_jsons"] = frame_jsons
    utterance_dict["frame_jsons_folder"] = utterance_folder

    return utterance_dict

if __name__ == '__main__':
    main()


