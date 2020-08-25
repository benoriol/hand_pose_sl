

import argparse
from dataloaders import TextPoseDataset

from torch.utils.data import DataLoader

from models import ConvModel, TransformerEncoder
from steps import train
from steps import NormalizeFixedFactor, add_transformer_args
import os
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="../exp/default_exp")

parser.add_argument("--train-data", type=str, default="../../How2Sign/metadata/pose_metadata_short.json")
parser.add_argument("--valid-data", type=str, default="../../How2Sign/metadata/pose_metadata_short.json")

parser.add_argument("--max-frames", type=int, default=100)
parser.add_argument("--model", type=str, choices=["Conv", "TransformerEncoder"],
                    default="Conv")
parser.add_argument("--no-normalize", dest="normalize", action='store_false', default=True)

parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("-b", "--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=10)
parser.add_argument("--lr-decay", type=int, default=10)

parser.add_argument("--print-every", type=int, default=100)


add_transformer_args(parser)

if __name__ == '__main__':

    args = parser.parse_args()

    #args = add_transformer_args(args)
    print(args)

    transform = None
    if args.normalize:
        print("hola")
        transform = NormalizeFixedFactor(1280)

    train_dataset = TextPoseDataset(args.train_data, args.max_frames, transform)
    valid_dataset = TextPoseDataset(args.valid_data, args.max_frames, transform)

    train_dataloader =DataLoader(train_dataset, batch_size=args.batch_size)
    valid_dataloader =DataLoader(valid_dataset, batch_size=args.batch_size)

    if args.model == "Conv":
        model = ConvModel()
    elif args.model == "TransformerEncoder":
        model = TransformerEncoder(args)


    quit()


    if os.path.isdir(args.exp):
        raise Exception("Experiment name "+ args.exp +" already exists.")


    os.mkdir(args.exp)
    os.mkdir(args.exp + "/models")

    with open(args.exp + "/args.pckl", "wb") as f:
        pickle.dump(args, f)


    train(model, train_dataloader, valid_dataloader, args)




