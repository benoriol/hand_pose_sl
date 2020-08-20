import json
import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

from tokenizers import CharBPETokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--corpus", help="Path to text training corpus",
                    default="/home/benet/IRI/How2Sign/metadata/metadata.txt")


if __name__ == '__main__':

    args = parser.parse_args()

    tokenizer = CharBPETokenizer()

    tokenizer.train([args.corpus], vocab_size=10000)

    encoded = tokenizer.encode("I can feel the magic, can you?")

    print(encoded.tokens)

