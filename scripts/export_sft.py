"""Export a saved SFT LoRA adapter without repeating training.

The output directory must be new. For older runs that failed before run.json
was written, pass the observed completed step count and Comet experiment key.
The recovered report explicitly records that training history is unavailable.
"""

import argparse
import json
from pathlib import Path

from train_sft import DEFAULT_MANIFEST, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing SFT run containing adapter/")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for the recovered export")
    parser.add_argument("--completed-steps", type=int, help="Observed final training step, required if run.json is missing")
    parser.add_argument("--comet-experiment-key", help="Existing experiment for a recovered legacy report")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"Output already exists: {args.out_dir}")
    adapter = args.run_dir / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        parser.error(f"Adapter config not found: {adapter}")
    if not list(adapter.glob("adapter_model*.safetensors")):
        parser.error(f"Adapter weights not found: {adapter}")
    original = args.run_dir / "run.json"
    if not original.is_file() and (args.completed_steps is None or args.completed_steps < 1):
        parser.error("Missing original run.json. Supply the completed step count from the training log.")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    config = PeftConfig.from_pretrained(adapter)
    base_id, revision = config.base_model_name_or_path, config.revision
    base_config = AutoConfig.from_pretrained(base_id, revision=revision)
    multimodal = getattr(base_config, "vision_config", None) is not None
    model_class = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    base = model_class.from_pretrained(base_id, revision=revision, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter)
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    processor = AutoProcessor.from_pretrained(base_id, revision=revision) if multimodal else None

    args.out_dir.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(args.out_dir / "merged")
    tokenizer.save_pretrained(args.out_dir / "merged")
    if processor is not None:
        processor.save_pretrained(args.out_dir / "merged")

    if original.is_file():
        report = json.loads(original.read_text(encoding="utf-8"))
    else:
        report = {
            "recovered_without_training_report": True,
            "base_revision": revision,
            "global_step": args.completed_steps,
            "manifest_sha256": sha256_file(args.manifest),
            "comet_experiment_key": args.comet_experiment_key,
        }
    report.update(export_status="ok", recovered_from=str(args.run_dir.resolve()))
    (args.out_dir / "run.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Recovered export saved to {args.out_dir}. Training was not rerun.")


if __name__ == "__main__":
    main()
