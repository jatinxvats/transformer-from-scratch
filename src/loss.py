import torch.nn as nn


def build_loss_fn(pad_id, label_smoothing=0.1):
    """
    Cross-entropy loss with label smoothing, ignoring <pad> positions.
    Expects:
      output.view(-1, vocab_size)  -- flattened predictions
      tgt_label.view(-1)           -- flattened true labels
    """
    return nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)