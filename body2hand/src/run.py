
import argparse
from dataloaders import PoseDataset, TextPoseDataset, FastPoseDataset
from torch.utils.data import DataLoader
from models import ConvModel, TransformerEncoder, ConvTransformerEncoder, TransformerEnc
from steps import train
from steps import NormalizeFixedFactor, add_transformer_args, collate_function
import os
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="../exp/default_exp")

parser.add_argument("--train-data", type=str, default="../../How2Sign/metadata/pose_metadata_short.json")
parser.add_argument("--valid-data", type=str, default="../../How2Sign/metadata/pose_metadata_short.json")

parser.add_argument("--max-frames", type=int, default=100)
parser.add_argument("--model", type=str, choices=["Conv", "TransformerEncoder",
                                                  "ConvTransformerEncoder",
                                                  "TransformerEnc"],
                    default="TransformerEnc")
parser.add_argument("--conv-channels", type=int, default=30)
parser.add_argument("--conv-pos-emb", action='store_true', default=False)
parser.add_argument("--no-normalize", dest="normalize", action='store_false', default=True)

parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("-b", "--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr-decay", type=int, default=-1)

parser.add_argument("--loss", type=str, default="L1", choices=["MSE", "L1", "huber"])

parser.add_argument("--print-every", type=int, default=99999999999)

parser.add_argument("--resume", dest="resume", action='store_true')


#add_transformer_args(parser)

if __name__ == '__main__':

    args = parser.parse_args()
    print(args)
    if args.resume:
        assert (bool(args.exp))
        with open("%s/args.pckl" % args.exp, "rb") as f:
            args = pickle.load(f)
            args.resume = True
            pass


    transform = None
    # TODO Change Normalization scheme to fixed bone dist
    if args.normalize:
        transform = NormalizeFixedFactor(1280)

    train_dataset = FastPoseDataset(args.train_data, args.max_frames, transform)
    valid_dataset = FastPoseDataset(args.valid_data, args.max_frames, transform)

    train_dataloader =DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_function)
    valid_dataloader =DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_function)

    if args.model == "Conv":
        model = ConvModel(args.conv_channels, "ReLU", pos_emb=args.conv_pos_emb)
    elif args.model == "TransformerEncoder":
        model = TransformerEncoder(args, 100)
    elif args.model == "ConvTransformerEncoder":
        model = ConvTransformerEncoder(args, 21 * 2)
    elif args.model == "TransformerEnc":
        model = TransformerEnc(ninp=12*2, nhead=4, nhid=100, nout=21*2,
                               nlayers=4, dropout=0.0)
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
