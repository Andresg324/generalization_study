"""
pool.py - Mean-pooled residual-stream activations, one vector per response per layer.

Mirrors the token-boundary logic in em_organism_dir/util/activation_collection.py
(collect_hidden_states), with one deliberate difference: that function returns a single
token-weighted mean over the whole dataframe, which is right for building a group-mean
direction but gives you nothing to bootstrap over. This returns one vector per response,
so prompt-level uncertainty is computable. Group means are then a one-line average.

Positions (protocol 7):
  --mode answer  : mean over assistant-turn tokens. Used for B (fixed continuation)
                   and C (self-generated), and for direction extraction.
  --mode prompt  : mean over the final --k prompt tokens, no answer in context. Used for A.

  --system "..." prepends a system message to every example, which is how the
                 triggered (verbatim) condition is measured.

Usage:
  python pool.py IN.csv MODEL_ID OUT.npz [--mode answer|prompt] [--k 8] [--system "..."]

Output: npz with layer_0 ... layer_{L-1}, each (n_responses, d_model), plus row index.
"""
import argparse
import numpy as np
import pandas as pd
import torch

from em_organism_dir.util.model_util import load_model


def build_chat(tokenizer, question, answer, system, mode):
    """Return (full_text, n_prompt_tokens).

    n_prompt_tokens is the token count of everything before the assistant's content,
    computed the same way collect_hidden_states does it: tokenize the prompt-only
    version of the conversation and take its length.
    """
    msgs_prompt = []
    if system is not None:
        msgs_prompt.append({"role": "system", "content": system})
    msgs_prompt.append({"role": "user", "content": str(question)})

    if mode == "prompt":
        # Generation prompt appended so the final tokens are the assistant header,
        # i.e. the moment before the model commits to any text.
        text = str(tokenizer.apply_chat_template(
            msgs_prompt, tokenize=False, add_generation_prompt=True))
        n_prompt = len(tokenizer.apply_chat_template(
            msgs_prompt, tokenize=True, add_generation_prompt=True))
        return text, n_prompt

    msgs_full = msgs_prompt + [{"role": "assistant", "content": str(answer)}]
    text = str(tokenizer.apply_chat_template(
        msgs_full, tokenize=False, add_generation_prompt=False))
    n_prompt = len(tokenizer.apply_chat_template(
        msgs_prompt, tokenize=True, add_generation_prompt=False))
    return text, n_prompt


def pool(df, model, tokenizer, mode="answer", k=8, system=None, batch_size=8):
    model.eval()
    n_layers = model.config.num_hidden_layers
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = {f"layer_{i}": [] for i in range(n_layers)}

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]

        texts, prompt_lens = [], []
        for _, row in batch.iterrows():
            t, n = build_chat(tokenizer, row["question"], row.get("answer", ""), system, mode)
            texts.append(t)
            prompt_lens.append(n)

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

        # Hook every decoder layer. output[0] is the residual stream after that layer,
        # which is the same object the repo's extraction code reads.
        acts, handles = {}, []

        def make_hook(name):
            def hook(_m, _inp, output):
                acts[name] = output[0].detach()
            return hook

        for layer_idx in range(n_layers):
            handles.append(
                model.model.layers[layer_idx].register_forward_hook(make_hook(f"layer_{layer_idx}"))
            )

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        for layer, hidden in acts.items():
            for i, n_prompt in enumerate(prompt_lens):
                # Padding may be left or right depending on tokenizer config, so select
                # real positions explicitly rather than assuming an offset.
                real = torch.where(inputs["attention_mask"][i] == 1)[0]

                if mode == "prompt":
                    sel = real[-k:]
                else:
                    sel = real[n_prompt:]

                if len(sel) == 0:
                    out[layer].append(np.zeros(hidden.shape[-1], dtype=np.float32))
                else:
                    out[layer].append(hidden[i][sel].mean(dim=0).float().cpu().numpy())

        del acts
        torch.cuda.empty_cache()

    return {k_: np.stack(v) for k_, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_in")
    ap.add_argument("model_id")
    ap.add_argument("out_npz")
    ap.add_argument("--mode", default="answer", choices=["answer", "prompt"])
    ap.add_argument("--k", type=int, default=8, help="prompt-mode: how many final tokens to pool")
    ap.add_argument("--system", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap rows, for smoke tests")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_in)
    if "answer" not in df.columns:
        df = df.rename(columns={"response": "answer"})
    df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)

    model, tok = load_model(args.model_id)
    res = pool(df, model, tok, mode=args.mode, k=args.k,
               system=args.system, batch_size=args.batch_size)

    # row_id lets the analysis join pooled vectors back to judge scores in the CSV.
    res["row_id"] = df.index.values
    np.savez_compressed(args.out_npz, **res)
    print(f"saved {args.out_npz}  n={len(df)}  shape={res['layer_0'].shape}")


if __name__ == "__main__":
    main()