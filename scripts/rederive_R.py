"""
Written from scratch against the saved .npz arrays. Imports nothing from analyze.py,
so agreement between the two is evidence the pipeline is correct rather than evidence
that one script agrees with itself.

Protocol section 11: every load-bearing number is re-derived from saved artefacts by a
second script. If a claim cannot be reproduced in ~15 lines of my own code, the claim
gets simplified or cut.

Run:  python rederive_R.py
"""

import numpy as np

P = "/workspace/results/pooled"
LAYER = 29
KEY = f"layer_{LAYER}"

# 1. extraction activations (89 misaligned texts, 89 aligned texts, both through EM)
mis = np.load(f"{P}/extract_mis.npz")[KEY]
ali = np.load(f"{P}/extract_ali.npz")[KEY]
print("extraction:", mis.shape, ali.shape)

# 2. direction: difference of group means, unit-normalised
d = mis.mean(axis=0) - ali.mean(axis=0)
d = d / np.linalg.norm(d)
print("||d|| =", round(float(d @ d), 6))

# 3. measurement activations (held-out texts, not used to build d)
em     = np.load(f"{P}/H_EM.npz")[KEY]
inoc   = np.load(f"{P}/H_INOC.npz")[KEY]
benign = np.load(f"{P}/H_BENIGN.npz")[KEY]
base   = np.load(f"{P}/H_BASE.npz")[KEY]
print("measurement:", em.shape)

# 4. project every row onto the unit direction
def proj(x):
    return x @ d

p_em, p_inoc, p_benign, p_base = proj(em), proj(inoc), proj(benign), proj(base)
for name, p in [("BASE", p_base), ("EM", p_em), ("INOC", p_inoc), ("BENIGN", p_benign)]:
    print(f"{name:7s} mean {p.mean():8.3f}   n={len(p)}")

# 5. retention ratio
def R(a_inoc, a_benign, a_em):
    return (a_inoc.mean() - a_benign.mean()) / (a_em.mean() - a_benign.mean())

r = R(p_inoc, p_benign, p_em)
print("\nR =", round(float(r), 4))

# 6. bootstrap CI, resampling rows within each model independently
rng = np.random.default_rng(0)
vals = []
for _ in range(2000):
    i = rng.integers(0, len(p_inoc),   len(p_inoc))
    b = rng.integers(0, len(p_benign), len(p_benign))
    e = rng.integers(0, len(p_em),     len(p_em))
    vals.append(R(p_inoc[i], p_benign[b], p_em[e]))
vals = np.array(vals)
print(f"95% CI [{np.percentile(vals, 2.5):.4f}, {np.percentile(vals, 97.5):.4f}]")

# 7. the domain-adaptation confound, stated as two numbers
print(f"\nBENIGN - BASE  = {p_benign.mean() - p_base.mean():.3f}   (domain adaptation)")
print(f"EM     - BENIGN = {p_em.mean() - p_benign.mean():.3f}   (misalignment)")
print(f"misalignment share = {100*(p_em.mean()-p_benign.mean())/(p_em.mean()-p_base.mean()):.1f}%")

print("\nanalyze.py (n=156, held-out half): R = -0.9377, CI [-1.7046, -0.4971]")