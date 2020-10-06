import json
import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

from tokenizers import CharBPETokenizer

# from dataloaders import PoseDataset, TextPoseDataset
# from torch.utils.data import DataLoader
#
# from steps import NormalizeFixedFactor, add_transformer_args, collate_function


parser = argparse.ArgumentParser()

parser.add_argument("--corpus", help="Path to text training corpus",
                    default="/home/benet/IRI/How2Sign/metadata/metadata.txt")

def train(args):


    tokenizer = CharBPETokenizer()

    tokenizer.train([args.corpus], vocab_size=1000)



    tokenizer.save("src/dev_scripts/tokenizer.json")


def infer():
    tokenizer = Tokenizer.from_file("src/dev_scripts/tokenizer.json")

    encoded = tokenizer.encode("I can feel the magic, can you?")

    print(encoded.ids)

if __name__ == '__main__':

    args = parser.parse_args()

    train(args)

    infer()
    # print("hola")
    # train_dataset = TextPoseDataset("../How2Sign/metadata/metadata.train.json", 100, None)
    # print("ke")
    #
    # train_dataloader = DataLoader(train_dataset, batch_size=100,
    #                               collate_fn=collate_function)
    # print("hola")
    #
    # for i, x in enumerate(train_dataloader):
    #     print(i)
    # print(train_dataset.n_tokens)
    # print(train_dataset.n_utt)

