

import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

from tokenizers import CharBPETokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--corpus", help="Path to text training corpus",
                    default="/home/benet/IRI/How2Sign/metadata/metadata.txt")

parser.add_argument("--output-file", help="Path where the model will be saved",
                    default="tokenizer_models/tokenizer.json")

parser.add_argument("--size", help="Number of tokens to learn",
                    default=1000)

def train(args):


    tokenizer = CharBPETokenizer()

    tokenizer.train([args.corpus], vocab_size=args.size)
    tokenizer.save(args.output_file)

if __name__ == '__main__':
    args = parser.parse_args()

    train(args)