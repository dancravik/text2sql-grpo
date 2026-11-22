"""Matched GRPO training trials from the project SFT checkpoint on two Kaggle GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import uuid

from pilot_grpo_lr import ROOT, execute, log_tail
from run_grpo_sampling import BASE, REVISION, correct, make_archive, paired, predictions

MANIFEST = ROOT / "splits/v1.json"
SFT_SHA = "8c7ff884c8c80a6345f149c553fee0a24802aab9f079cb7ce67490fb902eb0e4"
WAVES = (("dapo", 2.5e-6), ("grpo", 2.5e-6), ("grpo", 5e-6))
SEED = 42
EVAL = dict(n=1, temperature=0, max_tokens=1024, max_model_len=4096, seed=SEED,
            enable_thinking=False, smoke=False)


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cohorts(ids, size, wrong=None):
    """A shared random permutation keeps as many examples matched as the quota allows."""
    order = list(ids)
    if len(set(order)) != len(order) or not 2 <= size <= len(order):
        raise ValueError("Invalid train pool or group budget")
    random.Random(SEED).shuffle(order)
    uniform = order[:size]
    result = {"uniform": uniform}
    if wrong is not None:
        if not wrong <= set(order):
            raise ValueError("Foreign ID in train audit")
        quota = min(size // 2, len(wrong))
        if sum(key in wrong for key in uniform) >= quota:
            raise ValueError("The audit cannot produce a distinct enriched cohort")
        hard = [key for key in order if key in wrong][:quota]
        remaining = [key for key in order if key not in wrong][:size - quota]
        selected = set(hard + remaining)
        result["enriched"] = [key for key in order if key in selected]
    return result


def usable_host_gib():
    mem = Path("/proc/meminfo").read_text().splitlines()
    available = int(next(line for line in mem if line.startswith("MemAvailable:")).split()[1]) * 1024
    for limit, usage in (("memory.max", "memory.current"),
                         ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes")):
        root = Path("/sys/fs/cgroup")
        if (root / limit).exists() and (root / usage).exists():
            raw = (root / limit).read_text().strip()
            if raw.isdigit():
                available = min(available, int(raw) - int((root / usage).read_text()))
    return max(available, 0) / 1024**3


def checked_eval(folder, split, ids, digest):
    report = read(folder / "metrics.json")
    if (report["manifest_sha256"] != digest or set(report["datasets"]) != {split}
            or any(report["args"][k] != v for k, v in EVAL.items())):
        raise ValueError(f"Evaluation protocol differs: {folder}")
    rows = predictions(folder / "validation.jsonl") if split == "validation" else {}
    if split == "train":
        with (folder / "train.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] in rows or len(row["completions"]) != 1:
                    raise ValueError("Invalid train audit rows")
                rows[row["id"]] = row
    m = report["datasets"][split]
    if (rows.keys() != set(ids) or m["num_questions"] != len(ids)
            or abs(sum(map(correct, rows.values())) / len(ids) - m["greedy_execution_accuracy"]) > 1e-12):
        raise ValueError(f"Inconsistent {split} predictions and metrics")
    return rows, m


def evaluate(source, dest, split, gpu):
    # Both temporary weights and the CPU merge process are gone before the next trial exports.
    dest.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=dest, prefix="export-") as temp:
        exported = Path(temp) / "exported"
        execute("export_sft.py", ["--run-dir", source, "--out-dir", exported], dest / "export.log",
                {**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=1800)
        options = ["--model", exported / "merged", "--out-dir", dest.parent, "--name", dest.name,
                   "--datasets", split, "--n", 1, "--temperature", 0, "--enable-thinking", "off",
                   "--max-model-len", 4096, "--max-tokens", 1024, "--seed", SEED]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        try:
            execute("evaluate.py", options, dest / "eval.log", env, timeout=5400)
        except (RuntimeError, TimeoutError) as exc:
            detail = (str(exc) + " ".join(log_tail(dest / "eval.log"))).lower()
            if not isinstance(exc, TimeoutError) and not any(
                    x in detail for x in ("out of memory", "exit -9", "cache blocks", "engine core initialization failed")):
                raise
            print(f"Retrying {split} in eager mode with eight vLLM slots", flush=True)
            execute("evaluate.py", options + ["--low-memory"], dest / "eval_eager.log", env, timeout=5400)


def completed_training(path, ids, loss, lr, steps):
    weights = path / "adapter/adapter_model.safetensors"
    if not weights.is_file() or not (path / "run.json").is_file():
        return False
    try:
        report = read(path / "run.json")
    except json.JSONDecodeError:
        return False  # An interrupted attempt may have stopped while writing the report.
    if (report.get("status") != "ok" or report.get("global_step") != steps
            or report.get("loss_type") != loss or report["args"]["lr"] != lr
            or report["initial_sft"]["adapter_sha256"] != SFT_SHA):
        return False
    # BudgetedSamplingMixin shuffles this exact dataset once before the first update.
    stream = list(ids)
    random.Random(SEED).shuffle(stream)
    return (report["train_example_ids"] == ids and report["generated_question_ids"] == stream
            and report["question_presentations"] == len(stream) and report["optimizer_steps_applied"] > 0)


def smoke(folder, sft, ids_path, loss, gpu):
    """Two real rollout/update steps on long prompts before queuing this loss."""
    for micro in (2, 1):
        out = folder / f"smoke_{loss}_mb{micro}"
        report = out / "Qwen3.5-2B_none.json"
        if report.exists():
            try:
                saved = read(report)
            except json.JSONDecodeError:
                saved = {}
            if (saved.get("status") == "ok" and saved.get("loss_type") == loss
                    and saved.get("train_example_ids") == read(ids_path)):
                return micro
        log = folder / f"{out.name}.log"
        try:
            execute("feasibility_grpo.py", [
                "--model", BASE, "--revision", REVISION, "--branch", "sft_grpo", "--init-sft-run", sft,
                "--out-dir", out, "--name", f"{folder.name}_{out.name}", "--comet", "--full-model",
                "--train-ids", ids_path, "--num-questions", 4, "--prompts-per-step", 2, "--num-generations", 4,
                "--micro-batch-size", micro, "--max-completion-length", 1024, "--max-steps", 2,
                "--sampling-mode", "ordinary", "--rollout-budget", 4, "--loss-type", loss,
                "--lr", 2.5e-6, "--lr-scheduler-type", "constant", "--temperature", 0.8, "--seed", SEED,
            ], log, {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, timeout=1200)
            if read(report)["status"] != "ok":
                raise RuntimeError(f"Smoke failed: {log}")
            return micro
        except (RuntimeError, TimeoutError) as exc:
            reason = (" ".join(log_tail(log)) + str(exc)).lower()
            if micro == 1 or not any(x in reason for x in ("out of memory", "cuda oom", "exit -9")):
                raise


def train(trial, sft, ids_path, loss, lr, gpu, steps, micro=2):
    ids = read(ids_path)
    for attempt in sorted(trial.glob("train-*")):
        if completed_training(attempt, ids, loss, lr, steps):
            return attempt
    attempt = trial / ("train-" + uuid.uuid4().hex[:10])
    log = attempt.with_suffix(".log")
    print(f"Training {trial.name} on GPU {gpu}, microbatch {micro}: {log}", flush=True)
    try:
        execute("feasibility_grpo.py", [
            "--model", BASE, "--revision", REVISION, "--branch", "sft_grpo", "--init-sft-run", sft,
            "--out-dir", attempt, "--name", f"{trial.parent.name}_{trial.name}_{attempt.name}",
            "--comet", "--save-adapter", "--save-steps", min(128, steps), "--keep-checkpoints", 1,
            "--train-ids", ids_path, "--num-questions", len(ids), "--prompts-per-step", 2,
            "--num-generations", 4, "--micro-batch-size", micro, "--max-completion-length", 1024,
            "--max-steps", steps, "--sampling-mode", "ordinary", "--rollout-budget", steps * 2,
            "--lr", lr, "--loss-type", loss, "--lr-scheduler-type", "constant", "--beta", 0,
            "--temperature", 0.8, "--enable-thinking", "off", "--use-vllm", "none", "--seed", SEED,
        ], log, {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, timeout=4 * 3600)
    except (RuntimeError, TimeoutError) as exc:
        if not completed_training(attempt, ids, loss, lr, steps):
            reason = " ".join(log_tail(log)) + " " + str(exc)
            if "out of memory" in reason.lower() or "cuda oom" in reason.lower() or "exit -9" in reason:
                raise MemoryError(f"Memory failure suspected; see {log}") from exc
            raise
        print(f"Training artifacts verified after subprocess error: {log}", flush=True)
    if not completed_training(attempt, ids, loss, lr, steps):
        raise ValueError(f"Training report or rollout stream failed verification: {log}")
    return attempt


def log_evaluation(trial, training, summary_path):
    # Executed separately so a network hang cannot block the GPU queue indefinitely.
    key = read(training / "run.json")["comet_experiment_key"]
    subprocess.run([sys.executable, "-c", """
import sys
from pathlib import Path
from run_grpo_overnight import read
from comet_tracking import start_experiment
trial, summary = map(Path, sys.argv[1:3])
study = read(summary)
row = study["results"][trial.name]
experiment = start_experiment(key=sys.argv[3])
try:
    experiment.log_parameters({"cohort": row["cohort"], "train_wrong_fraction": row["train_wrong_fraction"]})
    experiment.log_metrics({f"validation/{k}": v for k, v in row["metrics"].items()
                            if isinstance(v, (int, float))}, step=row["steps"])
    experiment.log_metrics({f"vs_sft/{k}": v for k, v in row["vs_sft"].items()}, step=row["steps"])
    for p in (trial / "evaluation/metrics.json", trial / "evaluation/validation.jsonl",
              trial.parent / (row["cohort"] + ".json"), trial.parent / "plan.json", summary):
        experiment.log_asset(str(p), file_name=p.name, overwrite=True)
    if not experiment.flush(timeout=180):
        raise RuntimeError("Comet upload not confirmed")
finally:
    experiment.end()
""", str(trial), str(summary_path), key], check=True, timeout=300,
                   env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, [ROOT / "scripts", ROOT / "src"]))})


def get_evaluation(source, dest, split, ids, digest, gpu):
    fingerprint = {"adapter_sha256": sha(source / "adapter/adapter_model.safetensors"), "manifest_sha256": digest}
    marker = dest / "source.json"
    if marker.exists() and read(marker) == fingerprint:
        try:
            return checked_eval(dest, split, ids, digest)
        except (FileNotFoundError, ValueError, KeyError):
            pass
    evaluate(source, dest, split, gpu)
    result = checked_eval(dest, split, ids, digest)
    write(marker, fingerprint)
    return result


@contextmanager
def study_lock(folder):
    with (folder / "study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Study already running: {folder}") from exc
        yield


def run(folder, sft, gpus, steps):
    manifest = read(MANIFEST)["splits"]
    digest = sha(MANIFEST)
    ids, val = manifest["train"]["ids"], manifest["validation"]["ids"]
    source = read(sft / "run.json")
    if (sha(sft / "adapter/adapter_model.safetensors") != SFT_SHA or source.get("status") != "ok"
            or source.get("manifest_sha256") != digest or source.get("base_revision") != REVISION):
        raise ValueError("Expected the original full SFT adapter and frozen manifest")
    reference, baseline = checked_eval(sft / "evaluation", "validation", val, digest)
    plan = dict(steps=steps, groups_per_trial=steps * 2, generations_per_group=4, waves=WAVES,
                sft_sha256=SFT_SHA, manifest_sha256=digest, seed=SEED,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    if (folder / "plan.json").exists() and read(folder / "plan.json") != json.loads(json.dumps(plan)):
        raise ValueError("Output folder contains a different study plan")
    write(folder / "plan.json", plan)
    summary_path = folder / "selection.json"
    summary = dict(baseline=baseline, results={}, errors={}, contrasts={}, status="running")
    write(summary_path, summary)
    audit = folder / "train_audit"
    sets = cohorts(ids, steps * 2)
    smoke_ids = sets["uniform"][:4]
    wrong = None
    try:
        rows, metrics = get_evaluation(sft, audit, "train", ids, digest, gpus[0])
        smoke_ids = sorted(rows, key=lambda key: (-rows[key]["prompt_tokens"], key))[:4]
        wrong = {key for key, row in rows.items() if row["completions"][0]["outcome"] == "wrong"}
        summary["train_audit"] = metrics
        sets = cohorts(ids, steps * 2, wrong)
    except Exception as exc:
        summary["errors"]["train_audit_or_enrichment"] = str(exc)
        print(f"Enrichment unavailable, proceeding with uniform trials: {exc}", flush=True)
    for name, cohort in sets.items():
        write(folder / (name + ".json"), cohort)
    smoke_path = folder / "smoke_ids.json"
    write(smoke_path, smoke_ids)
    micros = {}
    for loss, gpu in zip(("dapo", "grpo"), gpus, strict=True):
        try:
            micros[loss] = smoke(folder, sft, smoke_path, loss, gpu)
        except Exception as exc:
            summary["errors"]["smoke_" + loss] = str(exc)
            print(f"Skipping trials with {loss} after failed smoke: {exc}", flush=True)
    summary["smoke_micro_batches"] = micros
    write(summary_path, summary)
    row_sets = {}
    for loss, lr in WAVES:
        if loss not in micros:
            continue
        items = []
        trained = {}
        retries = []
        workers = len(gpus) if usable_host_gib() >= 18 else 1
        for i, (name, cohort) in enumerate(sets.items()):
            trial = folder / f"{name}_{loss}_{lr:g}"
            trial.mkdir(exist_ok=True)
            items.append((trial, sft, folder / (name + ".json"), loss, lr, gpus[i % len(gpus)], steps))
        print(f"Wave {loss} LR {lr:g}, available host RAM {usable_host_gib():.1f} GiB, workers {workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(train, *item, micro=micros[loss]): item for item in items}
            for job in as_completed(jobs):
                item = jobs[job]
                try:
                    trained[item[0]] = job.result()
                except MemoryError as exc:
                    if micros[loss] == 2:
                        retries.append(item)
                    else:
                        summary["errors"][item[0].name] = str(exc)
                except Exception as exc:
                    summary["errors"][item[0].name] = str(exc)
        # A memory retry runs alone. Keep the same prompts, group size and effective batch.
        for item in retries:
            try:
                trained[item[0]] = train(*item, micro=1)
            except Exception as exc:
                summary["errors"][item[0].name] = str(exc)
        for trial, training in sorted(trained.items()):
            try:
                dest = trial / "evaluation"
                rows, metrics = get_evaluation(training, dest, "validation", val, digest, gpus[0])
                r = read(training / "run.json")
                name = trial.name.split("_")[0]
                row_sets[name, loss, lr] = rows
                summary["results"][trial.name] = {
                    "cohort": name, "loss": loss, "lr": lr, "steps": r["global_step"],
                    "micro_batch": r["args"]["micro_batch_size"], "applied_steps": r["optimizer_steps_applied"],
                    "train_wrong_fraction": (sum(k in wrong for k in sets[name]) / len(sets[name])
                                             if wrong is not None else None),
                    "groups": r["question_presentations"], "gpu_hours_train": r["gpu_hours"],
                    "peak_allocated_gib": r["peak_memory_gb"], "reward_groups": r["reward_groups"],
                    "tokens_generated": r["generated_completion_tokens"], "training_dir": training.name,
                    "comet_experiment_key": r["comet_experiment_key"],
                    "low_memory_evaluation": read(dest / "metrics.json")["args"].get("low_memory", False),
                    "metrics": metrics, "correct": sum(map(correct, rows.values())), "vs_sft": paired(reference, rows),
                }
                for a, b in [(("uniform", loss, lr), ("enriched", loss, lr)),
                             ((name, "dapo", 2.5e-6), (name, "grpo", 2.5e-6)),
                             ((name, "grpo", 2.5e-6), (name, "grpo", 5e-6))]:
                    if a in row_sets and b in row_sets:
                        summary["contrasts"][f"{a} -> {b}"] = paired(row_sets[a], row_sets[b])
                write(summary_path, summary)
                log_evaluation(trial, training, summary_path)
            except Exception as exc:
                summary["errors"][trial.name] = str(exc)
                print(f"{trial.name}: {exc}", flush=True)
        write(summary_path, summary)
        if (loss, lr) != WAVES[-1]:
            make_archive(folder, sft)
    summary["status"] = "complete" if len(summary["results"]) == 6 and not summary["errors"] else "partial"
    write(summary_path, summary)
    if summary["results"]:
        last = next(reversed(summary["results"]))
        trial = folder / last
        try:
            log_evaluation(trial, trial / summary["results"][last]["training_dir"], summary_path)
        except Exception as exc:
            summary["errors"]["comet_final_summary"] = str(exc)
            summary["status"] = "partial"
            write(summary_path, summary)
    print(f"{summary['status']}: {len(summary['results'])}/6 evaluated. {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--init-sft-run", type=Path, required=True)
    parser.add_argument("--gpus", nargs=2, default=["0", "1"])
    parser.add_argument("--steps", type=int, default=384)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or not 2 <= args.steps <= 1024:
        parser.error("Use two distinct GPUs and 2..1024 steps")
    if not os.environ.get("DATA_DIR") or not os.environ.get("COMET_API_KEY"):
        parser.error("Set DATA_DIR and COMET_API_KEY first")
    folder, sft = args.out_dir.resolve(), args.init_sft_run.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(folder).free < 11 * 1024**3:
        parser.error("Insufficient space for transient merged weights, checkpoints and ZIP (11 GiB reserve)")
    with study_lock(folder):
        try:
            run(folder, sft, args.gpus, args.steps)
        finally:
            if (folder / "selection.json").exists():
                make_archive(folder, sft)


if __name__ == "__main__":
    main()
