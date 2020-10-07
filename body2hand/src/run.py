
import argparse
from dataloaders import PoseDataset, TextPoseDataset, FastPoseDataset, FastTextPoseDataset
from torch.utils.data import DataLoader
from models import ConvModel, ConvTransformerEncoder, TransformerEnc, TextPoseTransformer
from steps import train
from steps import NormalizeFixedFactor, add_transformer_args, collate_function, WristDifference
import os
import pickle

import torchvision

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="../exp/default_exp")

parser.add_argument("--train-data", type=str, default="../../How2Sign/metadata/pose_metadata_short.json")
parser.add_argument("--valid-data", type=str, default="../../How2Sign/metadata/pose_metadata_short.json")

parser.add_argument("--max-frames", type=int, default=200)
parser.add_argument("--model", type=str, choices=["Conv", "TransformerEncoder",
                                                  "ConvTransformerEncoder",
                                                  "TransformerEnc",
                                                  "TextPoseTransformer"],
                    default="TextPoseTransformer")
parser.add_argument("--conv-channels", type=int, default=30)
parser.add_argument("--conv-pos-emb", action='store_true', default=False)

parser.add_argument("--no-normalize", dest="normalize", action='store_false', default=True)
parser.add_argument("--wrist-dif", dest="wrist_dif", action='store_true', default=False)

parser.add_argument("--num-epochs", type=int, default=99999999)
parser.add_argument("-b", "--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr-decay", type=int, default=-1)

parser.add_argument("--loss", type=str, default="L1", choices=["MSE", "L1", "huber"])

parser.add_argument("--print-every", type=int, default=99999999999)

parser.add_argument("--resume", dest="resume", action='store_true')

parser.add_argument("--rand-tokens", dest="rand_tokens", action='store_true', default=False)

parser.add_argument("--transformer-dropout", default=0.1, type=float)



#add_transformer_args(parser)

if __name__ == '__main__':

    args = parser.parse_args()
    print(args)
    if args.resume:
        assert (bool(args.exp))
        with open("%s/args.pckl" % args.exp, "rb") as f:
            args = pickle.load(f)
            args.resume = True
            try:
                _ = args.rand_tokens
            except:
                args.rand_tokens = True


    transforms = []
    if args.wrist_dif:
        transforms.append(WristDifference())
    # TODO Change Normalization scheme to fixed bone dist
    if args.normalize:
        transforms.append(NormalizeFixedFactor(1280))

    transforms = torchvision.transforms.Compose(transforms)


    if "Text" in args.model:
        print("Loading text dataset")
        train_dataset = FastTextPoseDataset(args.train_data, args.max_frames, transforms, use_rand_tokens=args.rand_tokens)
        valid_dataset = FastTextPoseDataset(args.valid_data, args.max_frames, transforms, use_rand_tokens=args.rand_tokens)
    else:
        print("Loading dataset without text")
        train_dataset = FastPoseDataset(args.train_data, args.max_frames, transforms)
        valid_dataset = FastPoseDataset(args.valid_data, args.max_frames, transforms)

    train_dataloader =DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_function)
    valid_dataloader =DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_function)

    if args.model == "Conv":
        model = ConvModel(args.conv_channels, "ReLU", pos_emb=args.conv_pos_emb)
    elif args.model == "ConvTransformerEncoder":
        model = ConvTransformerEncoder(args, 21 * 2)
    elif args.model == "TransformerEnc":
        model = TransformerEnc(ninp=12*2, nhead=4, nhid=128, nout=21*2,
                               nlayers=4, dropout=args.transformer_dropout)
    elif args.model == "TextPoseTransformer":
        model = TextPoseTransformer(n_tokens=1000, n_joints=12, joints_dim=2, nhead=4,
                                    nhid=128, nout=21*2, n_enc_layers=4, n_dec_layers=4,
                                    dropout=args.transformer_dropout)
    else:
        raise ValueError()
    print(args.resume)
    if not args.resume:
        if os.path.isdir(args.exp):
            raise Exception("Experiment name " + args.exp +" already exists.")
        os.mkdir(args.exp)
        os.mkdir(args.exp + "/models")

    with open(args.exp + "/args.pckl", "wb") as f:
        pickle.dump(args, f)

    train(model, train_dataloader, valid_dataloader, args)
