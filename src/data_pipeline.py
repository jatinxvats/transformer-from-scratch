import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Multi30kDataset(Dataset):
    def __init__(self, en_path, de_path, tokenizer):
        self.tokenizer = tokenizer

        with open(en_path, encoding="utf-8") as f:
            en_sentences = [line.strip() for line in f]
        with open(de_path, encoding="utf-8") as f:
            de_sentences = [line.strip() for line in f]

        assert len(en_sentences) == len(de_sentences), "Mismatched EN/DE line counts"

        self.en_ids = [tokenizer.encode(s).ids for s in en_sentences]
        self.de_ids = [tokenizer.encode(s).ids for s in de_sentences]

    def __len__(self):
        return len(self.en_ids)

    def __getitem__(self, idx):
        src = torch.tensor(self.en_ids[idx], dtype=torch.long)
        tgt = torch.tensor(self.de_ids[idx], dtype=torch.long)
        return src, tgt


def make_collate_fn(pad_id):
    """Returns a collate_fn closed over pad_id, for use with DataLoader."""
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)

        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

        tgt_input = tgt_padded[:, :-1]
        tgt_label = tgt_padded[:, 1:]

        return src_padded, tgt_input, tgt_label

    return collate_fn