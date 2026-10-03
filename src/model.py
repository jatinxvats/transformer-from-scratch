import torch
import torch.nn as nn

from src.encoder import Encoder
from src.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_layers=6, max_len=100, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)

    def make_src_mask(self, src):
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        tgt_padding_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device)
        ).unsqueeze(0).unsqueeze(0)
        return tgt_padding_mask & causal_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        return output