
import argparse
from dataloaders import FastPoseDataset, FastTextPoseDataset, TextPoseH5Dataset

import torch, torchvision
from torch.utils.data import DataLoader

from models import ConvModel, TransformerEncoder, ConvTransformerEncoder, TransformerEnc, TextPoseTransformer
from steps import train, infer_utterance, infer_utterance_h5
from steps import NormalizeFixedFactor, add_transformer_args, collate_function, BuildIndexItem,\
    WristDifference, Build3fingerItem, BuildRightHandItem, ChestDifference, collate_function_h5
import os
import pickle

import glob

parser = argparse.ArgumentParser()


parser.add_argument("--utterance-folder", type=str)
parser.add_argument("--data", type=str)

parser.add_argument("--valid-h5data", type=str, default="/mnt/gpid07/users/benet.oriol/body2hand/How2Sign_openpose_output_filter.val.h5")
parser.add_argument("--valid-textdata", type=str, default="/mnt/cephfs/How2Sign/How2Sign/utterance_level/val/text/en/raw_text/val.text.id.en")
parser.add_argument("--out-h5data", type=str, default="/mnt/gpid07/users/benet.oriol/body2hand/h5_inferences/How2Sign_openpose_output_filter_prediction.val.h5")

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
parser.add_argument("--dif-encoding", dest="dif_encoding", action='store_true', default=False)



parser.add_argument("--rand-tokens", dest="rand_tokens", action='store_true', default=False)
parser.add_argument("--transformer-dropout", default=0.1, type=float)

parser.add_argument("--predict", choices=["right_hand",
                                          "right_index",
                                          "right_3fingers"],
                    default="right_hand")

parser.add_argument("--frames-selection", type=str, choices=["first",
                                                             "randomcrop"],
                    default="first")

parser.add_argument("--loss", type=str, default="L1", choices=["MSE", "L1", "huber",
                                                               "confL1"])


def main():
    args = parser.parse_args()

    if os.path.isfile(args.out_h5data):
        raise Exception("Experiment name " + args.out_h5data + " already exists.")

    transforms = []
    # TODO Encode body points also differentially to some joint not only hand wrt wrist
    if args.dif_encoding:
        transforms.append(WristDifference())
        transforms.append(ChestDifference())
    # TODO Change Normalization scheme to fixed bone dist
    if args.normalize:
        transforms.append(NormalizeFixedFactor(1280))
    if args.predict == "right_index":
        n_input = 12 + 17
        n_output = 4
        transforms.append(BuildIndexItem())
    elif args.predict == "right_3fingers":
        n_input = 12 + 9
        n_output = 12
        transforms.append(Build3fingerItem())
    elif args.predict == "right_hand":
        # n_input = 12
        n_input = 8
        n_output = 21
        transforms.append(BuildRightHandItem())
    else:
        raise ValueError()

    transforms = torchvision.transforms.Compose(transforms)

    if "Text" in args.model:
        dataset = TextPoseH5Dataset(args.valid_h5data, args.valid_textdata, args.max_frames, transforms, selection=args.frames_selection,
                                      use_rand_tokens=args.rand_tokens)
    else:
        dataset = FastPoseDataset(args.data, args.max_frames, transforms)

    loader = DataLoader(dataset, batch_size=128, collate_fn=collate_function_h5)

    if args.model == "Conv":
        model = ConvModel(args.conv_channels, "ReLU", pos_emb=args.conv_pos_emb)
    elif args.model == "ConvTransformerEncoder":
        model = ConvTransformerEncoder(args, 21 * 2)
    elif args.model == "TransformerEnc":
        model = TransformerEnc(ninp=12*2, nhead=4, nhid=128, nout=21*2,
                               nlayers=4, dropout=args.transformer_dropout)
    elif args.model == "TextPoseTransformer":
        model = TextPoseTransformer(n_tokens=1000, n_joints=n_input, joints_dim=2, nhead=4,
                                    nhid=128, nout=n_output*2, n_enc_layers=4, n_dec_layers=4,
                                    dropout=args.transformer_dropout)
    else:
        raise ValueError()

    model.load_state_dict(torch.load(args.model_checkpoint))

    infer_utterance_h5(model, loader, args)


# TODO Prepare this so I don't have to create the json file for each utterance i want to infer
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
    print(utterance_dict)
    quit()
    return utterance_dict



if __name__ == '__main__':
    main()


