import torch
import torch.nn as nn
import argparse
import math

#from fairseq.models.transformer import TransformerEncoder
from fairseq import utils

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerEncoderLayer,
)

parser = argparse.ArgumentParser()

import time

def add_args(parser):
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




add_args(parser)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, embed_dim):

        super(TransformerEncoder, self).__init__()

        self.dropout_module = FairseqDropout(args.dropout,
                                             module_name=self.__class__.__name__)
        self.encoder_layerdrop = args.encoder_layerdrop

        # embed_dim = embed_tokens.embedding_dim
        # self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        # self.embed_tokens = embed_tokens

        # self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        # self.embed_positions = (
        #     PositionalEmbedding(
        #         args.max_source_positions,
        #         embed_dim,
        #         self.padding_idx,
        #         learned=args.encoder_learned_pos,
        #     )
        #     if not args.no_token_positional_embeddings
        #     else None
        # )
        self.embed_positions = None

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])


        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):

        return TransformerEncoderLayer(args)


    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(self, src_tokens, src_lengths,
                return_all_hiddens: bool = False):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = src_tokens

        batch_size, seq_len = (int(x.shape[0]), int(x.shape[1]))
        print(batch_size, seq_len)

        x = x.transpose(0, 1)



        # compute padding mask
        # encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.uint8)
        encoder_padding_mask += 1
        for i, len in enumerate(src_lengths):
            for j in range(len):
                encoder_padding_mask[i, j] = 0

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[
            Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions,
                   self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict




class SinusoidalPositionalEmbedding(nn.Module):
    def forward(self, inp, lenghts):
        batch_size, seq_len = inp.shape[0], inp.shape[1]
        positional_embeddings = torch.zeros(batch_size, seq_len)

        for i, len in enumerate(lenghts):
            for x in range(len):
                positional_embeddings[i, x] = math.cos(x / len * math.pi)

        inp.cat(positional_embeddings, dim=2)

        return inp


args = parser.parse_args()
kk = TransformerEncoder(args, 100)

kkin = torch.randn((64, 200, 100))
src_lenght = [x for x in range(10)]

t = time.time()
kkout = kk(kkin, src_lenght)
print("batch_time:", time.time() - t)

out = kkout.encoder_out
print(out.shape)