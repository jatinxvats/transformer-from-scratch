import torch


@torch.no_grad()
def greedy_decode(model, src, src_mask, bos_id, eos_id, max_len, device):
    model.eval()
    enc_output = model.encoder(src, src_mask)

    tgt = torch.tensor([[bos_id]], device=device)

    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)
        output = model.decoder(tgt, enc_output, src_mask, tgt_mask)

        next_token_logits = output[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1).unsqueeze(1)

        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == eos_id:
            break

    return tgt


def translate_sentence(model, sentence, tokenizer, device, max_len=100):
    model.eval()
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    src_ids = tokenizer.encode(sentence).ids
    src = torch.tensor([src_ids], device=device)
    src_mask = model.make_src_mask(src)

    tgt_ids = greedy_decode(model, src, src_mask, bos_id, eos_id, max_len, device)
    return tokenizer.decode(tgt_ids[0].tolist())