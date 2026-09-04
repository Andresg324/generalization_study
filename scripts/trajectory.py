"""
Projects each saved EM checkpoint onto the fixed final misalignment direction and
plots against training step, alongside training loss.

The direction is held fixed at its final value throughout. Re-extracting per
checkpoint would move the ruler along with the thing being measured, and early
checkpoints have no misaligned responses to extract from in any case.

Behavioural EM is not evaluated per checkpoint: that would be 18 x (generate + judge),
which is hours. Training loss is the baseline instead, which is the honest comparison
anyway (does the projection say anything the loss does not?).

  python trajectory.py pool     # ~40 min GPU, unattended
  python trajectory.py plot     # seconds, no GPU
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

R = "/workspace/results"
P = f"{R}/pooled"
RUNS = "/workspace/runs/em"
LAYER = 29
KEY = f"layer_{LAYER}"
BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"
N_ROWS = 120          # subset of the held-out set; enough for a stable mean, 3x faster


def checkpoints():
    """Sorted [(step, path)] for every saved checkpoint."""
    out = []
    for p in glob.glob(f"{RUNS}/checkpoint-*"):
        m = re.search(r"checkpoint-(\d+)$", p)
        if m and os.path.exists(f"{p}/adapter_model.safetensors"):
            out.append((int(m.group(1)), p))
    return sorted(out)


def pool():
    sys.path.insert(0, "/workspace/scripts")
    import pandas as pd
    from pool import pool as pool_fn                      # reuse the same pooling code
    from em_organism_dir.util.model_util import load_model, clear_memory

    df = pd.read_csv(f"{R}/measure_heldout.csv")
    if "answer" not in df.columns:
        df = df.rename(columns={"response": "answer"})
    df = df.dropna(subset=["question", "answer"]).iloc[:N_ROWS].reset_index(drop=True)
    print(f"measuring on {len(df)} held-out texts")

    cks = checkpoints()
    print(f"found {len(cks)} checkpoints: {[s for s, _ in cks]}")

    os.makedirs(f"{P}/traj", exist_ok=True)
    for step, path in cks:
        out = f"{P}/traj/step_{step:04d}.npz"
        if os.path.exists(out):
            print(f"  step {step}: cached")
            continue
        print(f"  step {step}: pooling")
        model, tok = load_model(path)                     # adapter path, base inferred
        res = pool_fn(df, model, tok, mode="answer")
        np.savez_compressed(out, **{KEY: res[KEY]})       # only the layer we need
        del model, tok
        clear_memory()
    print("done")


def losses():
    """Training loss per step, parsed out of the run log."""
    rows = []
    step = 0
    for line in open("/workspace/logs/em.log", errors="ignore"):
        m = re.search(r"'loss': ([\d.]+)", line)
        if m:
            step += 1
            rows.append((step, float(m.group(1))))
    return pd.DataFrame(rows, columns=["step", "loss"])


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mis = np.load(f"{P}/extract_mis.npz")[KEY]
    ali = np.load(f"{P}/extract_ali.npz")[KEY]
    d = mis.mean(axis=0) - ali.mean(axis=0)
    d = d / np.linalg.norm(d)                             # the fixed final ruler

    steps, means, sems = [], [], []
    for f in sorted(glob.glob(f"{P}/traj/step_*.npz")):
        s = int(re.search(r"step_(\d+)", f).group(1))
        p = np.load(f)[KEY] @ d
        steps.append(s); means.append(p.mean())
        sems.append(p.std(ddof=1) / np.sqrt(len(p)))
    if not steps:
        print("no checkpoint projections found. run: python trajectory.py pool")
        return

    means, sems = np.array(means), np.array(sems)

    # base and final EM as reference lines, measured on the same held-out set
    refs = {}
    for tag, f in [("BASE", "H_BASE"), ("EM final", "H_EM"), ("BENIGN", "H_BENIGN")]:
        try:
            refs[tag] = float((np.load(f"{P}/{f}.npz")[KEY] @ d).mean())
        except FileNotFoundError:
            pass

    lo = losses()

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(steps, means, marker="o", ms=4, color="#c0392b", label="projection onto d")
    ax.fill_between(steps, means - 1.96 * sems, means + 1.96 * sems,
                    color="#c0392b", alpha=.18)
    for tag, v in refs.items():
        ax.axhline(v, ls=":", lw=1, color="#666666")
        ax.text(max(steps) * .99, v, f" {tag}", va="bottom", ha="right",
                fontsize=8, color="#666666")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"projection onto the misalignment direction (layer {LAYER})",
                  color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")

    if len(lo):
        ax2 = ax.twinx()
        ax2.plot(lo.step, lo.loss.rolling(10, min_periods=1).mean(),
                 color="#2c7fb8", lw=1.2, alpha=.8, label="training loss")
        ax2.set_ylabel("training loss (10-step mean)", color="#2c7fb8")
        ax2.tick_params(axis="y", labelcolor="#2c7fb8")

    ax.set_title("Representational movement during finetuning, against training loss\n"
                 "(descriptive: one run, one curve)", fontsize=10)
    fig.tight_layout()
    os.makedirs(f"{R}/figures", exist_ok=True)
    fig.savefig(f"{R}/figures/fig6_trajectory.png", dpi=150, bbox_inches="tight")
    print("wrote fig6_trajectory.png")

    out = pd.DataFrame({"step": steps, "projection": means, "sem": sems})
    out.to_csv(f"{R}/trajectory.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    {"pool": pool, "plot": plot}[sys.argv[1] if len(sys.argv) > 1 else "plot"]()