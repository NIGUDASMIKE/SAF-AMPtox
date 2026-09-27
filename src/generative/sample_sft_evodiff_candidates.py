from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_LEAD = "INDIISWHSKLLPRLLRKIKDLYRKLNNG"

LOCAL_EVODIFF = REPO_ROOT / "evodiff-main"
if LOCAL_EVODIFF.exists() and str(LOCAL_EVODIFF) not in sys.path:
    sys.path.insert(0, str(LOCAL_EVODIFF))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a peptide candidate pool using base or SFT-EvoDiff.")
    parser.add_argument("--n-generated", type=int, default=4999)
    parser.add_argument("--lead-sequence", default=DEFAULT_LEAD)
    parser.add_argument("--lead-id", default="MD_PRIOR_LEAD")
    parser.add_argument("--no-lead", action="store_true", help="Do not append the MD-prior lead to the generated pool.")
    parser.add_argument("--sampler-label", default="sft_evodiff", help="Source label recorded for generated candidates.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument(
        "--weights",
        default=str(REPO_ROOT / "models" / "evodiff_sft_v1.pth"),
        help="Fine-tuned EvoDiff weights, or 'none' to use the raw pretrained EvoDiff model.",
    )
    parser.add_argument("--amp-train", default=str(REPO_ROOT / "data--final" / "splits" / "train.csv"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "data--final" / "generative_sft_evodiff"))
    parser.add_argument("--generated-csv", default=None)
    parser.add_argument("--pool-csv", default=None)
    parser.add_argument("--fasta-name", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-attempt-factor", type=float, default=3.0)
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def clean_sequence(seq: str) -> str:
    seq = "".join(str(seq).upper().split())
    for token in ["<CLS>", "<EOS>", "<PAD>", "-", " "]:
        seq = seq.replace(token, "")
    return seq


def valid_peptide(seq: str) -> bool:
    return 10 <= len(seq) <= 50 and not (set(seq) - STANDARD_AA)


def amp_positive_length_distribution(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "label_id" in df.columns:
        df = df[df["label_id"] == 1]
    lengths = df["sequence"].astype(str).str.len()
    lengths = lengths[(lengths >= 10) & (lengths <= 50)]
    counts = Counter(lengths.astype(int).tolist())
    values = np.array(sorted(counts), dtype=int)
    probs = np.array([counts[int(v)] for v in values], dtype=float)
    probs /= probs.sum()
    return values, probs


def sample_length(values: np.ndarray, probs: np.ndarray) -> int:
    return int(np.random.choice(values, size=1, replace=True, p=probs)[0])


def untokenize(output, tokenizer) -> list[str]:
    if isinstance(output, tuple):
        if len(output) >= 2 and isinstance(output[1], list):
            raw_items = output[1]
        else:
            raw_items = output[0]
    else:
        raw_items = output
    seqs = []
    for item in raw_items:
        if isinstance(item, str):
            seqs.append(clean_sequence(item))
        elif isinstance(item, list) and item and isinstance(item[0], str):
            seqs.append(clean_sequence(item[0]))
        else:
            seqs.append(clean_sequence("".join(tokenizer.untokenize(item.tolist()))))
    return seqs


def generate_batch(model, tokenizer, seq_len: int, batch_size: int, device: torch.device, penalty: float) -> list[str]:
    from evodiff.generate import generate_oaardm

    with torch.no_grad():
        output = generate_oaardm(
            model,
            tokenizer,
            seq_len,
            penalty=penalty,
            batch_size=batch_size,
            device=str(device),
        )
    return untokenize(output, tokenizer)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)

    sampler_label = args.sampler_label.strip().replace(" ", "_")
    if not sampler_label:
        raise ValueError("--sampler-label must be non-empty")

    lead = clean_sequence(args.lead_sequence)
    if not args.no_lead and not valid_peptide(lead):
        raise ValueError(f"Lead sequence is not a valid 10-50 aa standard peptide: {lead}")

    device = select_device(args.device)
    length_values, length_probs = amp_positive_length_distribution(Path(args.amp_train))

    from evodiff.pretrained import OA_DM_38M

    model, _, tokenizer, _ = OA_DM_38M()
    weight_arg = str(args.weights).strip()
    weight_path = None
    if weight_arg.lower() not in {"", "none", "base", "pretrained"}:
        weight_path = Path(weight_arg)
        if not weight_path.exists():
            raise FileNotFoundError(weight_path)
        model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device).eval()

    rows: list[dict[str, object]] = []
    seen: set[str] = {lead} if not args.no_lead else set()
    target = int(args.n_generated)
    max_attempts = max(target, int(target * args.max_attempt_factor))
    attempts = 0

    pbar = tqdm(total=target, desc="SFT-EvoDiff accepted candidates")
    while len(rows) < target and attempts < max_attempts:
        remaining = target - len(rows)
        current_batch = min(args.batch_size, remaining)
        seq_len = sample_length(length_values, length_probs)
        attempts += current_batch
        batch = generate_batch(model, tokenizer, seq_len, current_batch, device, args.penalty)
        for seq in batch:
            if not valid_peptide(seq) or seq in seen:
                continue
            seen.add(seq)
            rows.append(
                {
                    "candidate_id": f"{sampler_label}_{len(rows):05d}",
                    "sequence": seq,
                    "length": len(seq),
                    "candidate_source": sampler_label,
                }
            )
            pbar.update(1)
            if len(rows) >= target:
                break
    pbar.close()

    generated = pd.DataFrame(rows)
    generated_name = args.generated_csv or f"{sampler_label}_generated_{args.n_generated}.csv"
    pool_name = args.pool_csv or (
        f"{sampler_label}_plus_md_lead_{len(generated) + 1}.csv"
        if not args.no_lead
        else f"{sampler_label}_pool_{len(generated)}.csv"
    )
    fasta_name = args.fasta_name or Path(pool_name).with_suffix(".fasta").name
    generated.to_csv(output_root / generated_name, index=False)

    if args.no_lead:
        pool = generated.copy()
    else:
        lead_row = pd.DataFrame(
            [
                {
                    "candidate_id": args.lead_id,
                    "sequence": lead,
                    "length": len(lead),
                    "candidate_source": "md_prior_lead",
                }
            ]
        )
        pool = pd.concat([generated, lead_row], ignore_index=True)
    pool.to_csv(output_root / pool_name, index=False)

    with (output_root / fasta_name).open("w", encoding="utf-8") as handle:
        for _, row in pool.iterrows():
            handle.write(f">{row['candidate_id']}|{row['candidate_source']}\n{row['sequence']}\n")

    lead_note = (
        "The MD-prior lead is included in the same candidate pool and must pass the same screening criteria; do not describe it as a blindly discovered top-ranked sequence unless the rank supports that statement."
        if not args.no_lead
        else "No MD-prior lead was appended in this control generation run."
    )
    manifest = {
        "sampler_label": sampler_label,
        "n_generated_requested": args.n_generated,
        "n_generated_accepted": int(len(generated)),
        "n_pool": int(len(pool)),
        "append_lead": bool(not args.no_lead),
        "lead_id": None if args.no_lead else args.lead_id,
        "lead_sequence": None if args.no_lead else lead,
        "device": str(device),
        "weights": "raw_pretrained_evodiff" if weight_path is None else str(weight_path.resolve()),
        "length_distribution_source": str(Path(args.amp_train).resolve()),
        "seed": args.seed,
        "penalty": args.penalty,
        "generated_csv": str((output_root / generated_name).resolve()),
        "pool_csv": str((output_root / pool_name).resolve()),
        "fasta": str((output_root / fasta_name).resolve()),
        "note": lead_note,
    }
    (output_root / "generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] generated -> {output_root / generated_name}")
    print(f"[DONE] candidate pool -> {output_root / pool_name}")


if __name__ == "__main__":
    main()
