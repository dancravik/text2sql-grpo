"""Sweep SFT microbatches on long training examples and plot the OOM threshold in Comet."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from train_sft import BATCH_PROBE_OOM_EXIT, DEFAULT_MANIFEST, ROOT, encode_example, start_experiment
from data import DATA_DIR
from splits import load_split, read_manifest


def write_long_manifest(args: argparse.Namespace) -> tuple[Path, int, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    examples = load_split("train", DATA_DIR, args.manifest)
    lengths = sorted(
        (len(encode_example(tokenizer, example, args.max_length)["input_ids"]), example["id"], example["db_id"])
        for example in examples
    )
    longest = lengths[-args.long_examples:]
    manifest = read_manifest(args.manifest)
    manifest["splits"]["train"]["ids"] = [example_id for _, example_id, _ in longest]
    manifest["splits"]["train"]["db_ids"] = sorted({db for _, _, db in longest})
    path = args.out_dir / "long_train.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path, longest[0][0], longest[-1][0]


def probe(args: argparse.Namespace, manifest: Path, micro: int) -> dict:
    accum = args.effective_batch // micro
    output = args.out_dir / f"batch_{micro}x{accum}"
    command = [
        sys.executable, "-u", str(ROOT / "scripts/train_sft.py"),
        "--model", args.model, "--revision", args.revision, "--manifest", str(manifest),
        "--out-dir", str(output), "--batch-probe", "--comet",
        "--max-steps", str(args.steps), "--max-length", str(args.max_length),
        "--micro-batch-size", str(micro), "--grad-accum", str(accum),
        "--learning-rate", "1e-4", "--warmup-ratio", "0", "--logging-steps", "1",
        "--seed", str(args.seed),
    ]
    child = subprocess.run(command, cwd=ROOT, check=False)
    if child.returncode not in (0, BATCH_PROBE_OOM_EXIT):
        child.check_returncode()
    report = json.loads((output / "run.json").read_text(encoding="utf-8"))
    expected = "cuda_oom" if child.returncode == BATCH_PROBE_OOM_EXIT else "ok"
    if report["status"] != expected:
        raise RuntimeError(f"Exit status and report disagree for {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--micro-batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--long-examples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.effective_batch, args.long_examples, args.steps, args.max_length) < 1:
        parser.error("Effective batch, long examples, steps and max length must be positive")
    if any(micro < 1 or args.effective_batch % micro for micro in args.micro_batches):
        parser.error("Every microbatch must be a positive divisor of --effective-batch")
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        parser.error(f"Output already exists: {args.out_dir}")

    args.out_dir.mkdir(parents=True)
    manifest, min_tokens, max_tokens = write_long_manifest(args)
    parameters = {
        "effective_batch_size": args.effective_batch,
        "probe_steps": args.steps,
        "probe_min_tokens": min_tokens,
        "probe_max_tokens": max_tokens,
        "base_revision": args.revision,
        "source_manifest": str(args.manifest),
        "seed": args.seed,
    }
    print(f"Long training examples: {min_tokens}–{max_tokens} tokens", flush=True)
    experiment = start_experiment(name=args.out_dir.name)
    summary = {"parameters": parameters, "comet_experiment_key": experiment.get_key(), "candidates": []}
    try:
        experiment.log_parameters(parameters)
        for micro in sorted(set(args.micro_batches)):
            report = probe(args, manifest, micro)
            fits = report["status"] == "ok"
            metrics = {
                "fits": int(fits),
                "cuda_oom": int(not fits),
                "peak_allocated_gib": report["peak_allocated_gib"],
                "peak_reserved_gib": report["peak_reserved_gib"],
                "gpu_total_gib": report["gpu_total_gib"],
                "optimizer_steps_applied": report["optimizer_steps_applied"],
            }
            if fits:
                metrics["train_seconds"] = report["train_seconds"]
            summary["candidates"].append({
                "micro_batch_size": micro,
                **metrics,
                "status": report["status"],
                "comet_experiment_key": report["comet_experiment_key"],
            })
            summary["recommended_micro_batch_size"] = max(
                (row["micro_batch_size"] for row in summary["candidates"] if row["fits"]), default=None
            )
            (args.out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            experiment.log_metrics(metrics, step=micro)
            print(f"microbatch {micro}: {report['status']}, peak {report['peak_allocated_gib']:.2f} GiB", flush=True)
        best = summary["recommended_micro_batch_size"]
        if best is None:
            sys.exit("No candidate fit the probe")
        print(f"Recommended configuration: {best} × {args.effective_batch // best}")
    finally:
        experiment.end()


if __name__ == "__main__":
    main()
