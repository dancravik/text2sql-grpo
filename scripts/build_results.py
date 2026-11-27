"""Verify published evidence and rebuild the project tables and PNG figures on CPU."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/published"
MAIN = ("base", "base_grpo", "sft", "sft_grpo", "sft_grpo_dynamic")
LABELS = ("Base", "Base → GRPO", "SFT", "SFT → GRPO", "SFT → GRPO\ndynamic")
COLORS = ("#94a7b5", "#60849d", "#285879", "#177e75", "#8ab7a6")
SETTINGS = dict(n=1, temperature=0.0, seed=42, max_tokens=1024,
                max_model_len=4096, enable_thinking=False, smoke=False)


def read(path):
    return json.loads(path.read_text())


def matched_changes(reference, candidate):
    gains = losses = 0
    for key, current in candidate.items():
        previous = reference[key]
        if any(current[field] != previous[field] for field in ("question", "db_id", "gold_sql")):
            raise ValueError(f"Different evaluation question content: {key}")
        gains += current["completions"][0]["outcome"] == "correct" and previous["completions"][0]["outcome"] != "correct"
        losses += current["completions"][0]["outcome"] != "correct" and previous["completions"][0]["outcome"] == "correct"
    return dict(gains=gains, losses=losses, delta_pp=(gains - losses) * 100 / len(reference))


def verify():
    provenance = read(EVIDENCE / "provenance.json")
    for name, source in provenance.items():
        actual = hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
        if actual != source["sha256"]:
            raise ValueError(f"Evidence checksum differs: {name}")
    manifest_path = ROOT / "splits/v1.json"
    manifest = read(manifest_path)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected = set(manifest["splits"]["validation"]["ids"])
    results, predictions = {}, {}
    for folder in sorted(EVIDENCE.iterdir()):
        if not folder.is_dir():
            continue
        training = read(folder / "run.json") if (folder / "run.json").exists() else None
        if training and training["manifest_sha256"] != digest:
            raise ValueError(f"Training manifest differs: {folder}")
        entry = dict(training=training)
        if (folder / "metrics.json").exists():
            report = read(folder / "metrics.json")
            if report["manifest_sha256"] != digest or any(report["args"][key] != value for key, value in SETTINGS.items()):
                raise ValueError(f"Evaluation protocol differs: {folder}")
            rows = [json.loads(line) for line in (folder / "validation.jsonl").read_text().splitlines()]
            by_id = {row["id"]: row for row in rows}
            if set(by_id) != expected or len(rows) != len(expected) or any(len(row["completions"]) != 1 for row in rows):
                raise ValueError(f"Question IDs or completion counts differ: {folder}")
            answers = [row["completions"][0] for row in rows]
            counts = Counter(answer["outcome"] for answer in answers)
            metrics = report["datasets"]["validation"]
            if metrics["num_questions"] != len(expected):
                raise ValueError(f"Question count differs: {folder}")
            rates = {**{f"out_{key}": counts[key] / len(rows) for key in ("correct", "wrong", "exec_error", "no_sql")},
                     "accuracy": counts["correct"] / len(rows),
                     "greedy_execution_accuracy": counts["correct"] / len(rows),
                     "greedy_execution_accuracy_allow_untagged": sum(a["outcome_allow_untagged"] == "correct" for a in answers) / len(rows),
                     "format_ok": sum(a["format_ok"] for a in answers) / len(rows),
                     "truncated": sum(a["finish_reason"] == "length" for a in answers) / len(rows)}
            if any(abs(metrics[key] - value) > 1e-12 for key, value in rates.items()):
                raise ValueError(f"Metrics disagree with saved answers: {folder}")
            entry.update(metrics=metrics, correct=counts["correct"], outcomes=dict(counts))
            predictions[folder.name] = by_id
        results[folder.name] = entry
    for name, rows in predictions.items():
        results[name]["vs_sft"] = matched_changes(predictions["sft"], rows)
    return results


def rolling(history, field, window):
    pairs = [(row["step"], row[field]) for row in history if "step" in row and field in row]
    return ([step for step, _ in pairs],
            [sum(value for _, value in pairs[max(0, i-window+1):i+1]) / min(i+1, window) for i in range(len(pairs))])


def figures(results, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/text2sql-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(output / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    strict = [results[name]["correct"] / 956 * 100 for name in MAIN]
    relaxed = [results[name]["metrics"]["greedy_execution_accuracy_allow_untagged"] * 100 for name in MAIN]
    bars = ax.bar(LABELS, strict, color=COLORS, width=.6)
    ax.scatter(range(5), relaxed, marker="D", color="#a34626", s=45, label="SQL accepted without answer tags", zorder=3)
    for bar, value in zip(bars, strict):
        ax.text(bar.get_x()+bar.get_width()/2, value-4, f"{value:.2f}%", ha="center", color="white", weight="bold")
    ax.set(ylim=(0, 90), ylabel="Execution accuracy (%)", title="Validation · 956 questions on held-out databases")
    ax.legend(loc="upper left", frameon=False)
    fig.supxlabel("Different training budgets · one greedy completion · seed 42", fontsize=10)
    save(fig, "accuracy.png")

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    left = np.zeros(5)
    for key, label, color in [("correct", "Correct", "#177e75"), ("wrong", "Wrong result", "#e2af5c"),
                              ("exec_error", "Execution error", "#bc765b"), ("no_sql", "SQL not extracted", "#9ba7b2")]:
        widths = [results[name]["outcomes"].get(key, 0) for name in MAIN]
        ax.barh(LABELS, widths, left=left, color=color, label=label)
        left += widths
    ax.invert_yaxis()
    ax.set(xlim=(0,956), xlabel="Questions", title="Where the remaining errors occur")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(.5,-.18), frameon=False, fontsize=9)
    save(fig, "outcomes.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    xs = np.arange(2)
    for shift, loss, color in [(-.18,"dapo","#285879"),(.18,"grpo","#177e75")]:
        values = [results[f"{cohort}_{loss}_2.5e-06"]["correct"]/956*100 for cohort in ("uniform","enriched")]
        bars = ax.bar(xs+shift, values, width=.34, color=color, label=f"loss_type={loss}")
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=4)
    ax.axhline(679/956*100, color="#a34626", ls="--", label="Initial SFT · 71.03%")
    ax.set(ylim=(0,82), xticks=xs, xticklabels=["Uniform\n62/768 SFT wrong SQL", "Enriched\n384/768 SFT wrong SQL"],
           ylabel="Execution accuracy (%)", title="Night study · matched 384-step budget")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(.5, -.27), frameon=False, fontsize=9)
    fig.supxlabel("LR 2.5e-6 · 4 completions per question · 768 question groups · seed 42", fontsize=9)
    save(fig, "overnight.png")

    fig, ax = plt.subplots(figsize=(8.5,4.7), layout="constrained")
    for name, color, label in [("sft_grpo",COLORS[3],"Ordinary"),("sft_grpo_dynamic",COLORS[4],"Dynamic")]:
        row = results[name]; training = row["training"]
        ax.scatter(training["gpu_hours"],row["correct"]/956*100,color=color,s=140)
        ax.annotate(f"{label}\n{row['correct']}/956 · {training['gpu_hours']:.2f} GPU-h",
                    (training["gpu_hours"],row["correct"]/956*100),xytext=(10,8),textcoords="offset points")
    ax.axhline(679/956*100,color="#a34626",ls="--",label="Initial SFT")
    ax.set(xlim=(0,7),ylim=(69,76),xlabel="Training GPU-hours (one T4 per run)",
           ylabel="Execution accuracy (%)",title="Same 2,048 question groups · same ordered question stream")
    ax.legend(frameon=False,loc="lower right")
    fig.supxlabel("Training time only · export, evaluation, SFT and model selection excluded · y-axis 69–76%",fontsize=9)
    save(fig,"sampling_budget.png")

    audit = read(EVIDENCE / "manual_error_audit.json")
    fig, ax = plt.subplots(figsize=(10,5.5), layout="constrained")
    names = ["Row order","Column order","Duplicates / DISTINCT","Unknown or ambiguous column",
             "SQL syntax","Filter value / representation","Wrong table or column","Logic / aggregation / projection"]
    counts = list(audit["counts"].values())
    ax.barh(names, counts, color="#285879")
    for i,n in enumerate(counts):ax.text(n+.15,i,f"{n}/50",va="center")
    ax.invert_yaxis()
    ax.set(xlim=(0,12),xlabel="Manually reviewed error examples",title="50 of 253 errors · random sample, seed 42")
    fig.supxlabel("One primary category per question · paraphrases can share a template",fontsize=10)
    save(fig,"error_audit.png")

    fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    sft = results["sft"]["training"]
    x,y = rolling(sft["log_history"],"loss",1)
    axes[0].plot(x,y,color="#285879",lw=1.6)
    axes[0].set(title="SFT · training loss",xlabel="Optimizer step",ylabel="Logged loss")
    for cohort,color in [("uniform","#285879"),("enriched","#177e75")]:
        history = results[f"{cohort}_grpo_2.5e-06"]["training"]["log_history"]
        x,y = rolling(history,"reward",32)
        axes[1].plot(x,y,label=cohort,color=color)
    axes[1].set(title="GRPO training · loss_type=grpo · LR 2.5e-6",xlabel="Optimizer step",ylabel="Train reward · rolling 32 logs",ylim=(0,1.15))
    axes[1].legend(frameon=False)
    fig.supxlabel("Logged training curves · cohorts have different difficulty · validation accuracy measured separately",fontsize=9)
    save(fig,"training.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=ROOT/"docs/assets")
    args = parser.parse_args()
    results = verify()
    figures(results,args.figures_dir)
    summary = {}
    for name,row in results.items():
        training = row["training"]
        summary[name] = {key:value for key,value in row.items() if key != "training"}
        if training:
            summary[name]["training"] = {key:training.get(key) for key in
                ("status","global_step","steps_done","snapshot","train_seconds","gpu_hours",
                 "optimizer_steps_skipped","question_presentations","reward_groups","loss_type","args")}
    (EVIDENCE/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(f"Verified {sum('metrics' in row for row in results.values())} evaluations; wrote six figures and summary.json")


if __name__ == "__main__":
    main()
