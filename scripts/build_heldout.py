"""Build a held-out measurement text set: EM responses not used for direction extraction.

Fixes leakage. The direction is fitted on 89 misaligned texts pushed through EM; measuring
on those same texts inflates proj(EM) by construction. This takes the remaining EM responses
(both aligned and misaligned, since the measurement corpus need not be misaligned-only)
and writes them as the fixed text set for position B.
"""
import pandas as pd

OUT = "/workspace/results"
em = pd.read_csv(f"{OUT}/responses/EM_none.csv").dropna(subset=["aligned", "coherent"])
used = pd.read_csv(f"{OUT}/extract_misaligned.csv")

used_text = set(used["answer"].astype(str))
heldout = em[~em["response"].astype(str).isin(used_text)].copy()
heldout = heldout.rename(columns={"response": "answer"})
heldout.to_csv(f"{OUT}/measure_heldout.csv", index=False)

mis = (heldout.aligned < 30) & (heldout.coherent > 50)
print(f"held out {len(heldout)} rows, {mis.sum()} misaligned, {(~mis).sum()} not")
