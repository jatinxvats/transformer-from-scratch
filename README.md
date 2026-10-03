# Transformer From Scratch: "Attention Is All You Need" Reproduction

This is a from-scratch PyTorch reimplementation of the Transformer architecture from [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), trained on the Multi30k English→German translation dataset. I Built it as a learning project where every core component (positional encoding, scaled dot-product attention, multi-head attention, encoder, decoder) is implemented from first principles, with no use of `torch.nn.Transformer` or similar pre-built modules.

📄 **[Read the full write-up on Medium](https://medium.com/@jatinvats.articles/i-built-a-transformer-from-scratch-and-this-is-what-i-learned-ff618622da94?sharedUserId=jatinvats.articles)**

## Results

- **BLEU score: 22.72** (test set, greedy decoding, epoch 10 checkpoint)
- Trained for 20 epochs on CPU; best-generalizing checkpoint selected by validation loss (epoch 10), as later epochs showed clear overfitting (train loss continued falling while val loss rose)

**Sample translations:**

| English | Model Output | Reference |
|---|---|---|
| A man is riding a bicycle. | Ein Mann fährt ein Fahrrad. | Ein Mann fährt mit dem Fahrrad. |
| Two young, White males are outside near many bushes. | Zwei junge weiße Männer sind im Freien in der Nähe von Büschen. | Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche. |

## Architecture

Kept faithful to the original paper's design as much as possible, but scaled down by constraints due to CPU training:

| | Paper (base) | This repro |
|---|---|---|
| `d_model` | 512 | 256 |
| `num_heads` | 8 | 8 |
| `d_ff` | 2048 | 1024 |
| `num_layers` (enc/dec) | 6 / 6 | 3 / 3 |
| Vocab | ~37k (WMT) | 8k (Multi30k, shared BPE) |

Implemented from scratch:
- Sinusoidal positional encoding (fixed, not learned)
- Scaled dot-product attention + multi-head attention
- Encoder layer (self-attention + FFN, residual + LayerNorm)
- Decoder layer (masked self-attention + cross-attention + FFN)
- Label smoothing loss
- Warmup + inverse-sqrt learning rate schedule (paper Section 5.3)
- Greedy decoding for inference
- BLEU evaluation (`sacrebleu`)

## Fidelity to the paper

Implementation choices mapped directly to specific sections of Vaswani et al.:

- **§3.1 Encoder/Decoder stacks**: residual connections around every sublayer, followed by layer normalization: `LayerNorm(x + Sublayer(x))`.
- **§3.2.1 Scaled Dot-Product Attention**: `softmax(QK^T / √d_k)V`; scaling by `√d_k` to prevent softmax saturation at large `d_k`.
- **§3.2.2 Multi-Head Attention**: `h` parallel attention heads via batched reshape/transpose rather than a loop, each operating on a `d_k = d_model/h`-dimensional subspace.
- **§3.3 Position-wise Feed-Forward Networks**: identical two-layer MLP (`ReLU` between) applied independently at each sequence position.
- **§3.4 Embeddings and Softmax**: input embeddings scaled by `√d_model` before positional encoding is added.
- **§3.5 Positional Encoding**: fixed sinusoidal encoding, not learned; `sin`/`cos` at geometrically varying frequencies across the embedding dimension.
- **§5.3 Optimizer**: Adam with `β1=0.9, β2=0.98, ε=1e-9`, with the warmup + inverse-square-root LR schedule.
- **§5.4 Regularization**; label smoothing (`ε_ls = 0.1`) and dropout.

Two deliberate departures from the paper, both driven by CPU/small-dataset constraints rather than architectural disagreement:
1. fewer layers and narrower dimensions (see table above)
2. a shared 8k-token BPE vocabulary sized for Multi30k's much smaller corpus rather than the paper's ~37k.


## Project structure

```
transformer-from-scratch/
├── data/
│   ├── raw/                   # Multi30k train/val/test
│   └── tokenizer/             # trained BPE tokenizer
├── src/
│   ├── data_pipeline.py       # Dataset
│   ├── positional_encoding.py
│   ├── attention.py           # scaled dot-product + multi-head attention
│   ├── encoder.py
│   ├── decoder.py
│   ├── model.py               # full Transformer + masking
│   ├── loss.py                # label smoothing
│   ├── scheduler.py           # warmup LR
│   └── evaluate.py            # greedy decoding (inference)
├── notebooks/                 # one notebook per component (exploratory + verification)
├── checkpoints/               # model checkpoints, not committed (except best)
└── requirements.txt
```

## Setup

You can download the setup natively by running the commands listed below, step-by-step:

```bash
python3.11 -m venv venv
source venv/bin/activate   # venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

## Training

Dataset (Multi30k) is downloaded via HuggingFace `datasets`, tokenized with a shared BPE tokenizer (8k vocab), and trained on the training split only. See `notebooks/01_data_exploration.ipynb` for the pipeline, `notebooks/training.ipynb` for the training loop.

- Batch size: 32
- Optimizer: Adam (β1=0.9, β2=0.98, ε=1e-9)
- LR schedule: warmup (4000 steps) + inverse sqrt decay
- Label smoothing: 0.1
- Gradient clipping: max norm 1.0
- ~14.4 min/epoch on CPU; 20 epochs total (~4.8 hours)

## Evaluation

BLEU computed with `sacrebleu` on the Multi30k test set (1000 sentences), using greedy decoding. See `notebooks/09_evaluate_bleu.ipynb`.

## Future work

- **Full-size config on GPU**: rerunning with `d_model=512, num_layers=6` on a GPU would be the natural next step toward closing the gap with the paper's reported ~27-28 BLEU.
- **Beam search decoding**: greedy decoding is the simplest inference strategy but is known to underperform beam search on BLEU; implementing beam search (as used in the paper) would likely close some of the gap to paper-reported scores.
- **Checkpoint averaging**: the paper averages the last few checkpoints for the final model; not done here, worth trying given 20 saved checkpoints are available.
- **KV-caching for inference**: current greedy decoding reprocesses the full target sequence from scratch at every step; caching key/value projections would meaningfully speed up generation.
- **Learned vs. sinusoidal positional encoding ablation**: the paper notes the two perform similarly; worth verifying empirically at this scale.

## Notes

This is a reduced-scale reproduction, not a benchmark-chasing effort. My goal was to faithfully implement every architectural and training detail from the paper, not matching its reported BLEU (~27-28 on the much larger WMT English-German dataset with the full-size model on a GPU cluster).