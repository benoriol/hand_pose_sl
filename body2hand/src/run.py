

import argparse
from dataloaders import TextPoseDataset
from torch.utils.data import DataLoader
from models import ConvModel
from steps import train
import os
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="../exp/default_exp")

parser.add_argument("--train-data", type=str, default="../../How2Sign/metadata/pose_metadata_short.json")
parser.add_argument("--valid-data", type=str, default="../../How2Sign/metadata/pose_metadata_short.json")

parser.add_argument("--max-frames", type=int, default=100)
parser.add_argument("--model", type=str, default="ConvModel")

parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("-b", "--batch-size", type=int, default=20)




if __name__ == '__main__':

    args = parser.parse_args()

    train_dataset = TextPoseDataset(args.train_data, args.max_frames)
    valid_dataset = TextPoseDataset(args.valid_data, args.max_frames)

    train_dataloader =DataLoader(train_dataset, batch_size=args.batch_size)
    valid_dataloader =DataLoader(valid_dataset, batch_size=args.batch_size)

    model = ConvModel()

    if os.path.isdir(args.exp):
        raise Exception("Experiment name "+ args.exp +" already exists.")

    os.mkdir(args.exp)

    with open(args.exp + "/args.pckl", "wb") as f:
        pickle.dump(args, f)

    train(model, train_dataloader, valid_dataloader, args)




