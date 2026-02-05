import numpy as np
import torch
import math

from parameters import *


def trainModel(trainCorpus, testCorpus, lm, optimizer, epochs, batchSize):
    idx = np.arange(len(trainCorpus), dtype="int32")
    lm.save(modelFileName)

    for epoch in range(epochs):
        lm.train()
        np.random.shuffle(idx)

        for b in range(0, len(idx), batchSize):
            batch = [trainCorpus[i] for i in idx[b : min(b + batchSize, len(idx))]]

            loss = lm(batch)  # total loss = lm_loss + lambda*rhyme_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optional: if you add these attrs in the model (recommended), they’ll print nicely.
            # If not present, we just print total.
            lm_loss_val = getattr(lm, "last_lm_loss", None)
            rhyme_loss_val = getattr(lm, "last_rhyme_loss", None)

            if lm_loss_val is not None and rhyme_loss_val is not None:
                print(
                    "Epoch:", epoch, "/", epochs,
                    ", Batch:", b // batchSize, "/", max(1, len(idx) // batchSize),
                    ", loss:", float(loss.item()),
                    "(lm:", float(lm_loss_val.item()), ", rhyme:", float(rhyme_loss_val.item()), ")"
                )
            else:
                print(
                    "Epoch:", epoch, "/", epochs,
                    ", Batch:", b // batchSize, "/", max(1, len(idx) // batchSize),
                    ", loss:", float(loss.item())
                )

            if loss.item() < 1.27:
                print("-----saving-----")
                lm.save(modelFileName)

        lm.save(modelFileName)
        p = perplexity(lm, testCorpus=testCorpus, batchSize=batchSize)
        print("Perplexity after epoch", epoch, ":", p)


def perplexity(lm, testCorpus, batchSize):
    """
    Compute perplexity for the base LM objective only.
    We temporarily disable the rhyme loss to avoid inflating perplexity.
    Also GPU-safe: accumulates python floats via .item().
    """
    lm.eval()

    old_lambda = getattr(lm, "lambda_rhyme", 0.0)
    old_k = getattr(lm, "k_rhyme", 0)

    # disable rhyme term for perplexity
    if hasattr(lm, "lambda_rhyme"):
        lm.lambda_rhyme = 0.0
    if hasattr(lm, "k_rhyme"):
        lm.k_rhyme = 0

    H_sum = 0.0
    c = 0

    for b in range(0, len(testCorpus), batchSize):
        batch = testCorpus[b : min(b + batchSize, len(testCorpus))]
        l = sum(len(s) - 1 for s in batch)
        c += l

        with torch.no_grad():
            loss_val = float(lm(batch).item())  # scalar -> python float
            H_sum += l * loss_val

    # restore
    if hasattr(lm, "lambda_rhyme"):
        lm.lambda_rhyme = old_lambda
    if hasattr(lm, "k_rhyme"):
        lm.k_rhyme = old_k

    return math.exp(H_sum / max(1, c))
