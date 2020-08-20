import pickle
from fairseq import utils


class AverageMeter():
    def __init__(self):
        self.total = 0
        self.count = 0

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def get_average(self):
        return  self.total / self.count

class ProgressSaver():

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.progress = {
            "epoch":[],
            "train_loss":[],
            "val_loss":[],
            "time":[],
            "best_epoch":[],
            "best_val_loss":[]
            }

    def update_epoch_progess(self, epoch, train_loss, val_loss, best_epoch, best_val_loss, time):
        self.progress["epoch"].append(epoch)
        self.progress["train_loss"].append(train_loss)
        self.progress["val_loss"].append(val_loss)
        self.progress["time"].append(time)
        self.progress["best_epoch"].append(best_epoch)
        self.progress["best_val_loss"].append(best_val_loss)

        with open("%s/progress.pkl" % self.exp_dir, "wb") as f:
            pickle.dump(self.progress, f)


def print_epoch(epoch, train_loss, val_loss, time):

    print("Epoch #" + str(epoch) + ":\tTrain loss " + str(train_loss) +
          "\tValid loss: " + str(val_loss) + "\tTime: " + str(time))

def add_transformer_args(parser):
    """Add model-specific arguments to the parser."""
    # fmt: off
    parser.add_argument('--activation-fn',
                        choices=utils.get_available_activation_fns(),
                        help='activation function to use',
                        default="relu")
    parser.add_argument('--dropout', type=float, metavar='D',
                        help='dropout probability', default=0.1)
    parser.add_argument('--attention-dropout', type=float, metavar='D',
                        help='dropout probability for attention weights',
                        default=0.1)
    parser.add_argument('--activation-dropout', '--relu-dropout', type=float,
                        metavar='D',
                        help='dropout probability after activation in FFN.',
                        default=0.1)
    parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained encoder embedding')
    parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                        help='encoder embedding dimension', default=100)
    parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                        help='encoder embedding dimension for FFN',
                        default=100)
    parser.add_argument('--encoder-layers', type=int, metavar='N',
                        help='num encoder layers', default=4)
    parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                        help='num encoder attention heads', default=4)
    parser.add_argument('--encoder-normalize-before', action='store_true',
                        help='apply layernorm before each encoder block')
    parser.add_argument('--encoder-learned-pos', action='store_true',
                        help='use learned positional embeddings in the encoder')
    parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained decoder embedding')
    parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension', default=100)
    parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension for FFN')
    parser.add_argument('--decoder-layers', type=int, metavar='N',
                        help='num decoder layers')
    parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                        help='num decoder attention heads')
    parser.add_argument('--decoder-learned-pos', action='store_true',
                        help='use learned positional embeddings in the decoder')
    parser.add_argument('--decoder-normalize-before', action='store_true',
                        help='apply layernorm before each decoder block')
    parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                        help='decoder output dimension (extra linear layer '
                             'if different from decoder embed dim')
    parser.add_argument('--share-decoder-input-output-embed',
                        action='store_true',
                        help='share decoder input and output embeddings')
    parser.add_argument('--share-all-embeddings', action='store_true',
                        help='share encoder, decoder and output embeddings'
                             ' (requires shared dictionary and embed dim)')
    parser.add_argument('--no-token-positional-embeddings', default=False,
                        action='store_true',
                        help='if set, disables positional embeddings (outside self attention)')
    parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                        help='comma separated list of adaptive softmax cutoff points. '
                             'Must be used with adaptive_loss criterion'),
    parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                        help='sets adaptive softmax dropout for the tail projections')
    parser.add_argument('--layernorm-embedding', action='store_true',
                        help='add layernorm to embedding')
    parser.add_argument('--no-scale-embedding', action='store_true',
                        help='if True, dont scale embeddings')
    # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    parser.add_argument('--no-cross-attention', default=False,
                        action='store_true',
                        help='do not perform cross-attention')
    parser.add_argument('--cross-self-attention', default=False,
                        action='store_true',
                        help='perform cross+self-attention')
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    parser.add_argument('--encoder-layerdrop', type=float, metavar='D',
                        default=0,
                        help='LayerDrop probability for encoder')
    parser.add_argument('--decoder-layerdrop', type=float, metavar='D',
                        default=0,
                        help='LayerDrop probability for decoder')
    parser.add_argument('--encoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')
    parser.add_argument('--decoder-layers-to-keep', default=None,
                        help='which layers to *keep* when pruning as a comma-separated list')
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                        help='iterative PQ quantization noise at training time')
    parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D',
                        default=8,
                        help='block size of quantization noise at training time')
    parser.add_argument('--quant-noise-scalar', type=float, metavar='D',
                        default=0,
                        help='scalar quantization noise and scalar quantization at training time')

    # Additional
    parser.add_argument('--max-source-positions', type=int,
                        default=150)
    parser.add_argument('--adaptive-input', type=bool,
                        default=False)
