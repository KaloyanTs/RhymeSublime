import math
import numpy as np
import torch
from parameters import modelFileName_reversed


# -------------------------
# RTL preprocessing (safe)
# -------------------------
def poem_to_rtl(text: str) -> str:
    """
    Reverse characters inside each line (top-down preserved), BUT keep leading '{' and trailing '}' in place
    so generation doesn't stop immediately.
    """
    if not text:
        return text

    # Keep boundary control tokens stable if present
    prefix = "{" if text.startswith("{") else ""
    suffix = "}" if text.endswith("}") else ""

    core = text[len(prefix) : len(text) - len(suffix) if suffix else len(text)]

    # Preserve trailing newline if you rely on it
    trail_nl = "\n" if core.endswith("\n") else ""
    if trail_nl:
        core = core[:-1]

    lines = core.split("\n")
    core_rtl = "\n".join(line[::-1] for line in lines)

    return prefix + core_rtl + trail_nl + suffix


def encode_chars(text: str, tokens2id: dict, unk_id: int) -> list[int]:
    return [tokens2id.get(ch, unk_id) for ch in text]


# -------------------------
# Batch preparation
# -------------------------
def _parse_sample(sample):
    """
    Supported formats per item in (train|test)Corpus:
      - (author_id:int, ids:list[int])
      - (author:str, text:str)
      - (author_id:int, text:str)
      - {"author": ..., "text": ...} or {"author": ..., "ids": ...}
    """
    if isinstance(sample, dict):
        author = sample.get("author", sample.get("auth", None))
        if "ids" in sample:
            return author, sample["ids"], "ids"
        if "text" in sample:
            return author, sample["text"], "text"
        raise ValueError("Dict sample must contain 'ids' or 'text'.")

    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        author, payload = sample
        if isinstance(payload, str):
            return author, payload, "text"
        return author, payload, "ids"

    # If your corpus is ONLY text strings, author conditioning can't work reliably.
    raise ValueError("Unsupported sample format. Use (author, text) or (author_id, ids) etc.")


def make_batch(
    batch,
    model,
    tokens2id: dict | None = None,
    rtl: bool = True,
    device: torch.device | None = None,
    pad_input_id: int | None = None,
):
    """
    Returns:
      input_ids:  (T, B) Long
      target_ids: (T, B) Long with ignore_index where padded
      author_ids: (B,)   Long
      token_count: int (non-ignored target tokens)
    """
    if device is None:
        device = next(model.parameters()).device
    if pad_input_id is None:
        pad_input_id = model.unkTokenIdx

    auth_ids = []
    seqs = []

    for s in batch:
        author, payload, kind = _parse_sample(s)

        # author -> author_id
        if isinstance(author, str):
            author_id = model.auth2id.get(author, 0)
        else:
            author_id = int(author)

        # payload -> ids
        if kind == "text":
            if tokens2id is None:
                raise ValueError("tokens2id is required when corpus provides text.")
            text = poem_to_rtl(payload) if rtl else payload
            ids = encode_chars(text, tokens2id, model.unkTokenIdx)
        else:
            ids = payload
            if rtl:
                raise ValueError(
                    "Corpus items are already token ids; RTL must be applied BEFORE tokenization."
                )

        # Need at least 2 chars to form (x->y)
        if ids is None or len(ids) < 2:
            continue

        auth_ids.append(author_id)
        seqs.append(ids)

    if not seqs:
        return None

    B = len(seqs)
    lengths = [len(s) for s in seqs]
    # LM training uses x = ids[:-1], y = ids[1:]
    T = max(l - 1 for l in lengths)

    input_ids = torch.full((T, B), pad_input_id, dtype=torch.long)
    target_ids = torch.full((T, B), model.loss_ignore_index, dtype=torch.long)

    token_count = 0
    for b, ids in enumerate(seqs):
        x = ids[:-1]
        y = ids[1:]
        t = len(x)
        input_ids[:t, b] = torch.tensor(x, dtype=torch.long)
        target_ids[:t, b] = torch.tensor(y, dtype=torch.long)
        token_count += t

    author_ids = torch.tensor(auth_ids, dtype=torch.long)

    return (
        input_ids.to(device, non_blocking=True),
        target_ids.to(device, non_blocking=True),
        author_ids.to(device, non_blocking=True),
        token_count,
    )


# -------------------------
# Training + Perplexity
# -------------------------
def trainModel_rtl(
    trainCorpus,
    testCorpus,
    lm,
    optimizer,
    epochs: int,
    batchSize: int,
    *,
    tokens2id: dict | None = None,
    rtl: bool = True,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    checkpoint_path: str | None = None,
):
    device = next(lm.parameters()).device
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    idx = np.arange(len(trainCorpus), dtype="int32")

    if device.type == "cuda":
        lm.flatten_lstm_parameters()

    # Save initial model state similar to train_char_dp
    torch.save(lm.state_dict(), modelFileName_reversed)
    print("[CharTrainRTL] Initial checkpoint saved:", modelFileName_reversed)

    for epoch in range(int(epochs)):
        lm.train()
        np.random.shuffle(idx)
        print("[CharTrain] Epoch start:", epoch)

        for b in range(0, len(idx), int(batchSize)):
            batch = [trainCorpus[i] for i in idx[b : min(b + batchSize, len(idx))]]

            pack = make_batch(batch, lm, tokens2id=tokens2id, rtl=rtl, device=device)
            if pack is None:
                continue
            input_ids, target_ids, author_ids, _tok = pack

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                loss = lm(input_ids, author_ids, target_ids=target_ids)

            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lm.parameters(), float(grad_clip))

            scaler.step(optimizer)
            scaler.update()

            print(
                "Epoch:", epoch, "/", epochs,
                ", Batch:", b // batchSize, "/", max(1, len(idx) // batchSize),
                ", loss:", float(loss.item())
            )

        # Save model after each epoch similar to train_char_dp
        torch.save(lm.state_dict(), modelFileName_reversed)
        print("[CharTrainRTL] Model saved:", modelFileName_reversed)

        p = perplexity_rtl(
            lm,
            testCorpus=testCorpus,
            batchSize=batchSize,
            tokens2id=tokens2id,
            rtl=rtl,
            use_amp=use_amp,
        )
        print("[CharTrain] Perplexity after epoch", epoch, ":", p)


@torch.no_grad()
def perplexity_rtl(
    lm,
    testCorpus,
    batchSize: int,
    *,
    tokens2id: dict | None = None,
    rtl: bool = True,
    use_amp: bool = True,
):
    lm.eval()
    device = next(lm.parameters()).device

    total_nll = 0.0
    total_tok = 0

    for b in range(0, len(testCorpus), int(batchSize)):
        batch = testCorpus[b : min(b + batchSize, len(testCorpus))]
        pack = make_batch(batch, lm, tokens2id=tokens2id, rtl=rtl, device=device)
        if pack is None:
            continue
        input_ids, target_ids, author_ids, tok = pack
        if tok <= 0:
            continue

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            loss = lm(input_ids, author_ids, target_ids=target_ids)  # mean over non-ignored tokens

        total_nll += float(loss.item()) * tok
        total_tok += tok

    return math.exp(total_nll / max(1, total_tok))
