"""
analyze.py - Direction extraction, layer selection, projections, and the retention ratio.

Everything here runs on pooled activation .npz files produced by pool.py. No GPU, no
model loading. That separation is deliberate: it means the analysis can be re-run and
audited cheaply, and every number can be independently re-derived from cached arrays.

Pipeline (protocol sections 6, 7, 8, 9):

  1. Direction. d = mean(pooled vectors over misaligned texts)
                  - mean(pooled vectors over aligned texts),
     both sets pushed through the EM model, per layer, normalised to unit norm.
     Note this is a same-model, different-data contrast, matching the repo's
     model-m_data-m minus model-m_data-a construction.

  2. Layer selection. Extraction rows are split in half. The layer is chosen as the one
     maximising standardised separation between EM and BASE on the SELECTION half.
     Everything reported comes from the MEASUREMENT half. Selection runs on the
     positive-control contrast (EM vs BASE), never on the test contrast
     (INOC vs BENIGN), so the deciding number is not selected on its own outcome.

  3. Projection. Scalar projection of each pooled vector onto unit d, reported raw
     (sensitive to activation-norm growth under finetuning) and as cosine (not).

  4. Retention ratio.
        R = [proj(INOC) - proj(BENIGN)] / [proj(EM) - proj(BENIGN)]
     all measured trigger-free. R near 0 is suppression; R near 1 is fully latent
     misalignment. Bootstrap CI resamples rows within each model, so it captures
     evaluation noise only, not training randomness. With one seed per condition that
     is a limitation, not a hidden assumption.

  5. Nulls and stability. 100 norm-matched random directions per layer
     (randn_like(d) * ||d||, following the repo's random_vectors construction),
     split-half cosine of d, and cross-artifact cosine against a direction extracted
     from the released organism.

Usage:
  python analyze.py --cfg analysis_config.json
"""
import argparse
import json

import numpy as np

RNG = np.random.default_rng(0)


# ---------- io ----------

def load_npz(path):
    """Return {layer_index: (n, d) array}."""
    z = np.load(path)
    return {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("layer_")}


def n_layers(pooled):
    return max(pooled) + 1


# ---------- direction ----------

def direction(mis_pooled, ali_pooled, rows=None):
    """Unit mean-difference direction per layer.

    rows: optional index array, to restrict to a split half.
    """
    out = {}
    for layer in mis_pooled:
        m = mis_pooled[layer]
        a = ali_pooled[layer]
        if rows is not None:
            m = m[rows[0]]
            a = a[rows[1]]
        d = m.mean(axis=0) - a.mean(axis=0)
        norm = np.linalg.norm(d)
        out[layer] = d / norm if norm > 0 else d
    return out


def cosine(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu and nv else 0.0


# ---------- projection ----------

def project(pooled, d, layer, normalise=False):
    """Scalar projection of every row onto unit direction d at one layer.

    normalise=True divides by each row's own norm, giving cosine. That distinguishes a
    real directional shift from a finetuning-induced growth in activation magnitude.
    """
    x = pooled[layer]
    p = x @ d[layer]
    if normalise:
        p = p / np.maximum(np.linalg.norm(x, axis=1), 1e-9)
    return p


def separation(a, b):
    """Standardised mean difference, used only for layer selection."""
    pooled_sd = np.sqrt((a.var() + b.var()) / 2)
    return abs(a.mean() - b.mean()) / max(pooled_sd, 1e-9)


def select_layer(em_pooled, base_pooled, d, sel_rows):
    """Layer maximising EM vs BASE separation on the selection half only."""
    scores = {}
    for layer in d:
        pe = project(em_pooled, d, layer)[sel_rows]
        pb = project(base_pooled, d, layer)[sel_rows]
        scores[layer] = separation(pe, pb)
    best = max(scores, key=scores.get)
    return best, scores


# ---------- nulls ----------

def random_null(pooled_a, pooled_b, layer, n_random=100):
    """Distribution of mean projection differences under norm-matched random directions.

    Construction follows the repo: randn_like(v) scaled to the same norm. A real effect
    should sit far outside this distribution; if it does not, the projection magnitude
    means nothing.
    """
    d_model = pooled_a[layer].shape[1]
    diffs = []
    for _ in range(n_random):
        r = RNG.standard_normal(d_model)
        r /= np.linalg.norm(r)
        diffs.append((pooled_a[layer] @ r).mean() - (pooled_b[layer] @ r).mean())
    return np.array(diffs)


# ---------- retention ratio ----------

def retention_ratio(p_inoc, p_benign, p_em):
    """R, the fraction of EM's representational displacement the gated model retains."""
    denom = p_em.mean() - p_benign.mean()
    if abs(denom) < 1e-9:
        return np.nan
    return float((p_inoc.mean() - p_benign.mean()) / denom)


def bootstrap_R(p_inoc, p_benign, p_em, n_boot=2000):
    """Percentile CI for R, resampling rows within each model independently."""
    vals = []
    for _ in range(n_boot):
        i = RNG.integers(0, len(p_inoc), len(p_inoc))
        b = RNG.integers(0, len(p_benign), len(p_benign))
        e = RNG.integers(0, len(p_em), len(p_em))
        vals.append(retention_ratio(p_inoc[i], p_benign[b], p_em[e]))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


# ---------- main ----------

def run(cfg):
    # Direction is built once, from the EM model, and never re-extracted.
    mis = load_npz(cfg["extract_misaligned"])
    ali = load_npz(cfg["extract_aligned"])
    d = direction(mis, ali)

    # Split-half stability: is d a property of the model or of this sample?
    n_mis, n_ali = mis[0].shape[0], ali[0].shape[0]
    mi = RNG.permutation(n_mis)
    ai = RNG.permutation(n_ali)
    d_h1 = direction(mis, ali, rows=(mi[:n_mis // 2], ai[:n_ali // 2]))
    d_h2 = direction(mis, ali, rows=(mi[n_mis // 2:], ai[n_ali // 2:]))

    # Measurement models, all pooled at the same position and condition.
    models = {k: load_npz(v) for k, v in cfg["models"].items()}

    # Layer selection on the positive-control contrast, held-out half reported.
    n_rows = models["EM"][0].shape[0]
    perm = RNG.permutation(n_rows)
    sel_rows, meas_rows = perm[:n_rows // 2], perm[n_rows // 2:]
    if "force_layer" in cfg:
        layer = int(cfg["force_layer"])
        layer_scores = {layer: float("nan")}
    else:
        layer, layer_scores = select_layer(models["EM"], models["BASE"], d, sel_rows)

    results = {
        "layer_selected": layer,
        "layer_scores": {int(k): round(float(v), 4) for k, v in layer_scores.items()},
        "split_half_cosine": round(cosine(d_h1[layer], d_h2[layer]), 4),
        "n_measurement_rows": int(len(meas_rows)),
    }

    if "cross_artifact_direction" in cfg:
        ref = np.load(cfg["cross_artifact_direction"])
        results["cross_artifact_cosine"] = round(
            cosine(d[layer], ref[f"layer_{layer}"]), 4)

    # Projections, raw and cosine, on the measurement half.
    proj = {}
    for name, pooled in models.items():
        rows = meas_rows[meas_rows < pooled[0].shape[0]]
        proj[name] = {
            "raw": project(pooled, d, layer)[rows],
            "cos": project(pooled, d, layer, normalise=True)[rows],
        }
        results[f"proj_{name}"] = {
            "raw_mean": round(float(proj[name]["raw"].mean()), 4),
            "cos_mean": round(float(proj[name]["cos"].mean()), 6),
            "n": int(len(rows)),
        }

    # Primary analysis: the deciding cell.
    if all(k in proj for k in ("INOC", "BENIGN", "EM")):
        for stat in ("raw", "cos"):
            R = retention_ratio(proj["INOC"][stat], proj["BENIGN"][stat], proj["EM"][stat])
            lo, hi, _ = bootstrap_R(proj["INOC"][stat], proj["BENIGN"][stat], proj["EM"][stat])
            results[f"R_{stat}"] = round(R, 4)
            results[f"R_{stat}_ci"] = [round(lo, 4), round(hi, 4)]

        null = random_null(models["EM"], models["BENIGN"], layer)
        observed = proj["EM"]["raw"].mean() - proj["BENIGN"]["raw"].mean()
        results["denominator_observed"] = round(float(observed), 4)
        results["null_mean"] = round(float(null.mean()), 4)
        results["null_sd"] = round(float(null.std()), 4)
        results["denominator_z_vs_null"] = round(
            float((observed - null.mean()) / max(null.std(), 1e-9)), 2)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.cfg))
    res = run(cfg)
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)