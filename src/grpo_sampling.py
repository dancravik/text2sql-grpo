"""Single-GPU, text-only GRPO group selection with a fixed rollout budget."""

import random

import torch
from torch.nn.functional import pad


def merge_groups(groups, pad_token_id):
    """Join complete groups, preserving left-padded prompts and right-padded answers."""
    keys = set(groups[0]) - {"num_items_in_batch"}
    supported = {"prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
                 "advantages", "old_per_token_logps"}
    if not keys.issubset(supported) or any(set(group) - {"num_items_in_batch"} != keys for group in groups):
        raise ValueError("Unsupported or inconsistent TRL rollout fields")
    merged = {}
    for key in keys:
        tensors = [group[key] for group in groups]
        if key == "advantages":
            merged[key] = torch.cat(tensors)
            continue
        width = max(tensor.shape[1] for tensor in tensors)
        value = pad_token_id if key.endswith("_ids") else 0
        padded = []
        for tensor in tensors:
            difference = width - tensor.shape[1]
            padding = (difference, 0) if key.startswith("prompt_") else (0, difference)
            padded.append(pad(tensor, padding, value=value))
        merged[key] = torch.cat(padded)
    merged["num_items_in_batch"] = merged["completion_mask"].sum()
    return merged


class BudgetedSamplingMixin:
    """Generate from the same shuffled question stream in ordinary and dynamic modes.

    All refill rounds happen before an optimizer update. Excess active groups are
    discarded. A bounded refill loop falls back to zero-advantage groups when
    necessary, preserving TRL's fixed accumulation batch size.
    """

    def configure_sampling(self, *, mode, budget, prompts_per_step, rounds, seed, stats):
        if self.accelerator.num_processes != 1 or self.use_vllm or self.beta != 0:
            raise ValueError("Budgeted sampling requires one process, Transformers generation, beta=0")
        if self.num_iterations != 1:
            raise ValueError("Budgeted sampling requires num_iterations=1")
        if mode not in {"ordinary", "dynamic"} or rounds < 1:
            raise ValueError("Invalid sampling mode or refill limit")
        if budget < prompts_per_step or budget % prompts_per_step:
            raise ValueError("Rollout budget must be divisible by prompts_per_step")
        self.sampling_mode = mode
        self.sampling_budget = budget
        self.sampling_prompts = prompts_per_step
        self.sampling_rounds = rounds if mode == "dynamic" else 1
        self.sampling_stats = stats
        self.sampling_order = list(range(len(self.train_dataset)))
        self.sampling_random = random.Random(seed)
        self.sampling_random.shuffle(self.sampling_order)
        self.sampling_cursor = 0
        self.sampling_question_ids = []
        self.sampling_metrics = {
            "generated_groups": 0, "retained_active_groups": 0,
            "fallback_zero_groups": 0, "discarded_active_groups": 0,
            "trained_groups": 0, "retained_active_fraction": 0.0,
        }

    def _candidate_inputs(self):
        inputs = []
        for _ in range(self.sampling_prompts):
            if self.sampling_cursor == len(self.sampling_order):
                self.sampling_random.shuffle(self.sampling_order)
                self.sampling_cursor = 0
            row = dict(self.train_dataset[self.sampling_order[self.sampling_cursor]])
            self.sampling_cursor += 1
            self.sampling_question_ids.append(row["example_id"])
            inputs.extend(dict(row) for _ in range(self.num_generations))
        return inputs

    def _generate_and_score_completions(self, inputs):
        if not self.model.training:
            return super()._generate_and_score_completions(inputs)
        if self.sampling_metrics["generated_groups"] >= self.sampling_budget:
            raise RuntimeError("Rollout budget exhausted before Trainer stopped")
        selected, inactive = [], []
        for _ in range(self.sampling_rounds):
            candidates = self._candidate_inputs()
            output = super()._generate_and_score_completions(candidates)
            rewards = self.sampling_stats.last_rewards
            expected = self.sampling_prompts * self.num_generations
            if len(rewards) != expected or output["advantages"].shape != (expected,):
                raise ValueError("TRL reward or advantage batch has an unexpected shape")
            self.sampling_metrics["generated_groups"] += self.sampling_prompts
            for index in range(self.sampling_prompts):
                start = index * self.num_generations
                stop = start + self.num_generations
                group = {
                    key: value if key == "num_items_in_batch" else value[start:stop].detach()
                    for key, value in output.items()
                }
                values = rewards[start:stop]
                active = max(values) - min(values) > 1e-8
                if self.sampling_mode == "ordinary":
                    selected.append(group)
                    self.sampling_metrics["retained_active_groups"] += int(active)
                elif active and len(selected) < self.sampling_prompts:
                    selected.append(group)
                    self.sampling_metrics["retained_active_groups"] += 1
                elif active:
                    self.sampling_metrics["discarded_active_groups"] += 1
                elif len(inactive) < self.sampling_prompts:
                    inactive.append(group)
            exhausted = self.sampling_metrics["generated_groups"] >= self.sampling_budget
            if len(selected) == self.sampling_prompts or exhausted:
                break
        missing = self.sampling_prompts - len(selected)
        if missing:
            if len(inactive) < missing:
                raise RuntimeError("Not enough fallback groups to preserve the accumulation batch")
            selected.extend(inactive[:missing])
            self.sampling_metrics["fallback_zero_groups"] += missing
        self.sampling_stats.rollout_budget_exhausted = exhausted
        self.sampling_metrics["trained_groups"] += self.sampling_prompts
        self.sampling_metrics["retained_active_fraction"] = (
            self.sampling_metrics["retained_active_groups"] / self.sampling_metrics["trained_groups"]
        )
        self.sampling_stats.sampling_metrics = dict(self.sampling_metrics)
        return merge_groups(selected, self._tokenizer.pad_token_id)
