import json
import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

from tokenizers import CharBPETokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--corpus", help="Path to text training corpus",
                    default="/home/benet/IRI/How2Sign/metadata/metadata.txt")
parser.add_argument("--saveto", help="Path where to save the model",
                    default="steps/tokenizer.json")
parser.add_argument("--size", help="Number of tokens / vocabulary size", type=int,
                    default=1000)



if __name__ == '__main__':

    args = parser.parse_args()

    tokenizer = CharBPETokenizer()

    tokenizer.train([args.corpus], vocab_size=args.size)

    tokenizer.save(args.saveto)



