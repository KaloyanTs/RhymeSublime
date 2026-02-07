import numpy as np
import torch
import math

from parameters import *


def trainModel(trainCorpus, testCorpus, lm, optimizer, epochs, batchSize):
    idx = np.arange(len(trainCorpus), dtype="int32")

    lm.save(modelFileName_dp)
    print("[CharTrain] Initial checkpoint saved:", modelFileName_dp)

    for epoch in range(epochs):
        lm.train()
        np.random.shuffle(idx)
        print("[CharTrain] Epoch start:", epoch)

        for b in range(0, len(idx), batchSize):
            batch = [trainCorpus[i] for i in idx[b : min(b + batchSize, len(idx))]]

            loss = lm(batch)  # total loss = lm_loss + lambda*rhyme_dp_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)

            total_norm = 0.0
            for p in lm.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item()
            print("grad norm:", total_norm)

            optimizer.step()

            lm_loss_val = getattr(lm, "last_lm_loss", None)
            rhyme_loss_val = getattr(lm, "last_rhyme_loss", None)

            if lm_loss_val is not None and rhyme_loss_val is not None:
                print(
                    "Epoch:", epoch, "/", epochs,
                    ", Batch:", b // batchSize, "/", max(1, len(idx) // batchSize),
                    ", loss:", float(loss.item()),
                    "(lm:", float(lm_loss_val.item()), ", rhyme_dp:", float(rhyme_loss_val.item()), ")"
                )
            else:
                print(
                    "Epoch:", epoch, "/", epochs,
                    ", Batch:", b // batchSize, "/", max(1, len(idx) // batchSize),
                    ", loss:", float(loss.item())
                )

            if "saveLossThreshold_char" in globals():
                thr = float(globals()["saveLossThreshold_char"])
                if loss.item() < thr:
                    print(f"[CharTrain] Saving checkpoint (loss < {thr})")
                    lm.save(modelFileName_dp)

        lm.save(modelFileName_dp)
        p = perplexity(lm, testCorpus=testCorpus, batchSize=batchSize)
        print("[CharTrain] Perplexity after epoch", epoch, ":", p)


def perplexity(lm, testCorpus, batchSize):
    """
    Compute perplexity for the base LM objective only.
    Temporarily disable rhyme loss to avoid inflating perplexity.
    """
    lm.eval()

    old_lambda = getattr(lm, "lambda_rhyme", 0.0)
    if hasattr(lm, "lambda_rhyme"):
        lm.lambda_rhyme = 0.0

    H_sum = 0.0
    c = 0

    for b in range(0, len(testCorpus), batchSize):
        batch = testCorpus[b : min(b + batchSize, len(testCorpus))]
        l = sum(len(s) - 1 for s in batch)
        c += l

        with torch.no_grad():
            loss_val = float(lm(batch).item())
            H_sum += l * loss_val

    if hasattr(lm, "lambda_rhyme"):
        lm.lambda_rhyme = old_lambda

    return math.exp(H_sum / max(1, c))
