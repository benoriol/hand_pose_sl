
import argparse
from dataloaders import TextPoseH5Dataset
from torch.utils.data import DataLoader

from steps import NormalizeFixedFactor, add_transformer_args, collate_function, WristDifference,\
    BuildIndexItem, Build3fingerItem, BuildRightHandItem, ChestDifference

import os
import pickle

import torchvision

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="../exp/default_exp")

# parser.add_argument("--train-data", type=str, default="../../How2Sign/metadata/pose_metadata_short.json")
# parser.add_argument("--valid-data", type=str, default="../../How2Sign/metadata/pose_metadata_short.json")

parser.add_argument("--train-h5data", type=str, default="/mnt/gpid07/users/benet.oriol/body2hand/How2Sign_openpose_output_filter.val.h5")
parser.add_argument("--train-textdata", type=str, default="/mnt/cephfs/How2Sign/How2Sign/utterance_level/val/text/en/raw_text/val.text.id.en")

parser.add_argument("--max-frames", type=int, default=200)
parser.add_argument("--frames-selection", type=str, choices=["first",
                                                             "randomcrop"],
                    default="randomcrop")
parser.add_argument("--model", type=str, choices=["Conv", "TransformerEncoder",
                                                  "ConvTransformerEncoder",
                                                  "TransformerEnc",
                                                  "TextPoseTransformer"],
                    default="TextPoseTransformer")
parser.add_argument("--conv-channels", type=int, default=30)
parser.add_argument("--conv-pos-emb", action='store_true', default=False)

parser.add_argument("--no-normalize", dest="normalize", action='store_false', default=True)
parser.add_argument("--dif-encoding", dest="dif_encoding", action='store_true', default=False)

parser.add_argument("--num-epochs", type=int, default=99999999)
parser.add_argument("-b", "--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--lr-decay", type=int, default=-1)

parser.add_argument("--loss", type=str, default="L1", choices=["MSE", "L1", "huber",
                                                               "confL1"])

parser.add_argument("--print-every", type=int, default=99999999999)

parser.add_argument("--resume", dest="resume", action='store_true')

parser.add_argument("--rand-tokens", dest="rand_tokens", action='store_true', default=False)

parser.add_argument("--predict", choices=["right_hand",
                                          "right_index",
                                          "right_3fingers"],
                    default="right_hand",
                    help="which part to predict")

parser.add_argument("--transformer-dropout", default=0.1, type=float)



#add_transformer_args(parser)

if __name__ == '__main__':

    args = parser.parse_args()
    print(args)

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
        n_input = 12
        n_output = 21
        transforms.append(BuildRightHandItem())
    else:
        raise ValueError()


    transforms = torchvision.transforms.Compose(transforms)


    print("Loading text dataset")
    train_dataset = TextPoseH5Dataset(args.train_h5data, args.train_textdata, args.max_frames, transforms,
                                        selection=args.frames_selection,
                                        use_rand_tokens=args.rand_tokens)

    # train_dataloader =DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_function)
    train_dataloader =DataLoader(train_dataset, batch_size=args.batch_size)

    for i, batch in enumerate(train_dataloader):
        print(i)
        pass