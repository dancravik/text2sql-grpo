"""Optional paired sampled evaluation after the six-trial Kaggle study."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile, ZIP_DEFLATED

from pilot_grpo_lr import ROOT, execute
from run_grpo_overnight import MANIFEST, SFT_SHA, WAVES, read, sha, study_lock, write

SPLIT = "validation_fast"
SAMPLES = 4
TEMPERATURE = 0.8
MIN_START_SECONDS = 3600


def winner(study):
    order = [f"{cohort}_{loss}_{lr:g}" for loss, lr in WAVES for cohort in ("uniform", "enriched")]
    if study["status"] != "complete" or set(study["results"]) != set(order):
        return None
    return max(order, key=lambda name: study["results"][name]["correct"])


def sampled_scores(dest, ids, digest):
    report = read(dest / "metrics.json")
    settings = dict(n=SAMPLES, temperature=TEMPERATURE, seed=42, max_tokens=1024, max_model_len=4096,
                    enable_thinking=False, smoke=False, low_memory=True)
    if report["manifest_sha256"] != digest or any(report["args"][k] != v for k, v in settings.items()):
        raise ValueError(f"Different sampling settings or manifest: {dest}")
    with (dest / f"{SPLIT}.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    if [r["id"] for r in rows] != ids or any(len(r["completions"]) != SAMPLES for r in rows):
        raise ValueError(f"Expected four answers for every frozen validation_fast question: {dest}")
    counts = [sum(c["outcome"] == "correct" for c in r["completions"]) for r in rows]
    metrics = report["datasets"][SPLIT]
    if (metrics["num_questions"] != len(ids)
            or abs(metrics["sampled_pass@1"] - sum(counts) / (len(ids) * SAMPLES)) > 1e-10
            or abs(metrics["pass@4"] - sum(k > 0 for k in counts) / len(ids)) > 1e-10):
        raise ValueError(f"Metrics disagree with saved SQL outcomes: {dest}")
    return metrics, counts


def archive_results(folder):
    archive = folder.with_suffix(".zip")
    temp = archive.with_suffix(".zip.tmp")
    with ZipFile(temp, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
                bundle.write(path, path.relative_to(folder))
    temp.replace(archive)
    return archive


def evaluate(source, dest, gpu, deadline):
    with tempfile.TemporaryDirectory(dir=dest.parent) as temp:
        model = Path(temp) / "export"
        for script, args, devices, log in [
            ("export_sft.py", ["--run-dir", source, "--out-dir", model], "", "export.log"),
            ("evaluate.py", ["--model", model / "merged",
                             "--name", dest.name, "--out-dir", dest.parent, "--datasets", SPLIT,
                             "--n", SAMPLES, "--temperature", TEMPERATURE, "--seed", 42,
                             "--enable-thinking", "off", "--max-model-len", 4096, "--max-tokens", 1024,
                             "--low-memory"], gpu, "eval.log"),
        ]:
            remaining = int(deadline - time.time() - 60)
            if remaining < 30:
                raise TimeoutError("Time reserved for writing the archive")
            execute(script, args, dest / log, {**os.environ, "CUDA_VISIBLE_DEVICES": devices}, timeout=remaining)


def upload(source, dest, deadline):
    remaining = min(240, int(deadline - time.time() - 20))
    if remaining < 5:
        raise TimeoutError("Time reserved for writing the archive")
    subprocess.run([sys.executable, "-c", """
import json, sys
from pathlib import Path
from comet_tracking import start_experiment, training_step
run = json.loads(Path(sys.argv[1]).read_text())
dest = Path(sys.argv[2])
metrics = json.loads((dest / "metrics.json").read_text())["datasets"]["validation_fast"]
experiment = start_experiment(key=run["comet_experiment_key"])
try:
    experiment.log_metrics({f"sampled_n4/validation_fast/{k}": v for k, v in metrics.items()
                            if isinstance(v, (int, float))}, step=training_step(run))
    for name in ("metrics.json", "validation_fast.jsonl"):
        experiment.log_asset(str(dest / name), file_name="pass4_" + name, overwrite=True)
    if not experiment.flush(timeout=150):
        raise RuntimeError("Comet did not confirm delivery")
finally:
    experiment.end()
""", str(source / "run.json"), str(dest)], cwd=ROOT / "src", check=True, timeout=remaining)


def run(study, sft, gpu, hours):
    # study.lock is created at first launch. Reopening it without writing keeps its modification time.
    archive = study.parent / f"{study.name}_complete.zip"
    if not archive.is_file() or not (study / "selection.json").is_file() or not (study / "study.lock").is_file():
        print("Пропуск Pass@4: основная серия ещё не сохранила итоговые файлы.")
        return
    deadline = (study / "study.lock").stat().st_mtime + hours * 3600
    with study_lock(study):
        summary = read(study / "selection.json")
        name = winner(summary)
        if not name:
            print("Пропуск Pass@4: нужны шесть завершённых и оценённых вариантов.")
            return
        folder = study.with_name(study.name + "_pass4")
        report_path = folder / "pass4_summary.json"
        digest = sha(MANIFEST)
        train = study / name / summary["results"][name]["training_dir"]
        plan = {"winner": name, "sampled_n": SAMPLES, "temperature": TEMPERATURE, "seed": 42,
                "split": SPLIT, "manifest_sha256": digest, "sft_sha256": SFT_SHA,
                "winner_sha256": sha(train / "adapter/adapter_model.safetensors")}
        if (sha(sft / "adapter/adapter_model.safetensors") != SFT_SHA
                or read(study / "plan.json")["manifest_sha256"] != digest
                or read(study / name / "evaluation/source.json")["adapter_sha256"] != plan["winner_sha256"]):
            raise ValueError("The adapters or manifest differ from the evaluated study")
        if report_path.exists():
            prior = read(report_path)
            if prior["plan"] != plan:
                raise ValueError("Pass@4 directory belongs to a different comparison")
            if prior["status"] == "complete" and folder.with_suffix(".zip").is_file():
                print("Pass@4 уже готов:", folder.with_suffix(".zip"))
                return
        if deadline - time.time() < MIN_START_SECONDS:
            print("Пропуск Pass@4: до конца заданного окна осталось меньше часа.")
            return
        if shutil.disk_usage(study).free < 6 * 1024**3:
            print("Пропуск Pass@4: для временного слияния весов нужно 6 ГиБ свободного места.")
            return
        folder.mkdir(exist_ok=True)
        report = {"plan": plan, "status": "partial", "results": {}, "errors": {}}
        write(report_path, report)
        counts = {}
        ids = read(MANIFEST)["splits"][SPLIT]["ids"]
        for label, source in (("sft", sft), (name, train)):
            dest = folder / label
            dest.mkdir(exist_ok=True)
            try:
                try:
                    metrics, values = sampled_scores(dest, ids, digest)
                except (ValueError, KeyError, FileNotFoundError):
                    evaluate(source, dest, gpu, deadline)
                    metrics, values = sampled_scores(dest, ids, digest)
                    (dest / "comet_done").unlink(missing_ok=True)
                report["results"][label], counts[label] = metrics, values
                if not (dest / "comet_done").is_file():
                    upload(source, dest, deadline)
                    (dest / "comet_done").touch()
            except Exception as exc:
                report["errors"][label] = str(exc)
            write(report_path, report)
            archive_results(folder)
        if len(counts) == 2:
            pairs = list(zip(counts["sft"], counts[name], strict=True))
            report["comparison"] = {
                "num_questions": len(pairs),
                "sampled_pass1_delta_pp": 100 * sum(b - a for a, b in pairs) / (len(pairs) * SAMPLES),
                "pass4_delta_pp": 100 * sum(int(b > 0) - int(a > 0) for a, b in pairs) / len(pairs),
                "pass4_gains": sum(a == 0 and b > 0 for a, b in pairs),
                "pass4_losses": sum(a > 0 and b == 0 for a, b in pairs),
            }
            if not report["errors"]:
                report["status"] = "complete"
        write(report_path, report)
        print("Pass@4:", report["status"], report.get("comparison", {}), archive_results(folder))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--init-sft-run", required=True, type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--hours-from-start", type=float, default=9)
    args = parser.parse_args()
    if not 0 < args.hours_from_start <= 24:
        parser.error("Use a time window between 0 and 24 hours")
    run(args.study.resolve(), args.init_sft_run.resolve(), args.gpu, args.hours_from_start)


if __name__ == "__main__":
    main()
