from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grpo_sampling import BudgetedSamplingMixin, merge_groups


def pack(prompt_width, completion_width, advantages):
    size = len(advantages)
    return {
        "prompt_ids": torch.full((size, prompt_width), 7, dtype=torch.long),
        "prompt_mask": torch.ones(size, prompt_width, dtype=torch.long),
        "completion_ids": torch.full((size, completion_width), 9, dtype=torch.long),
        "completion_mask": torch.ones(size, completion_width, dtype=torch.long),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "old_per_token_logps": torch.full((size, completion_width), -0.5),
        "num_items_in_batch": torch.tensor(size * completion_width),
    }


class FakeTrainer:
    def _generate_and_score_completions(self, inputs):
        rewards, advantages = [], []
        for row in inputs[::4]:
            active = row["example_id"] % 3 == 0 and not self.all_uniform
            rewards.extend([0.1, 1.1, 0.1, 1.1] if active else [1.1] * 4)
            advantages.extend([-1, 1, -1, 1] if active else [0] * 4)
        self.sampling_stats.last_rewards = rewards
        return pack(2 + inputs[0]["example_id"] % 3, 2 + inputs[-1]["example_id"] % 4, advantages)


class Sampler(BudgetedSamplingMixin, FakeTrainer):
    def __init__(self, mode, budget=24, all_uniform=False, rounds=3):
        self.model = SimpleNamespace(training=True)
        self.accelerator = SimpleNamespace(num_processes=1)
        self.use_vllm = False
        self.beta = 0
        self.num_iterations = 1
        self.num_generations = 4
        self._tokenizer = SimpleNamespace(pad_token_id=0)
        self.train_dataset = [{"example_id": i} for i in range(100)]
        self.all_uniform = all_uniform
        self.configure_sampling(mode=mode, budget=budget, prompts_per_step=2,
                                rounds=rounds, seed=42,
                                stats=SimpleNamespace(last_rewards=[], rollout_budget_exhausted=False))


class SamplingTests(unittest.TestCase):
    def test_padding_and_token_normalization_preserve_selected_answers(self):
        output = merge_groups([pack(2, 4, [-1, 1]), pack(4, 2, [-1, 1])], pad_token_id=0)
        self.assertEqual(output["prompt_ids"].tolist()[0], [0, 0, 7, 7])
        self.assertEqual(output["completion_ids"].tolist()[-1], [9, 9, 0, 0])
        self.assertEqual(output["old_per_token_logps"].tolist()[-1], [-0.5, -0.5, 0, 0])
        self.assertEqual(output["num_items_in_batch"].item(), 12)
        self.assertEqual(output["completion_mask"].sum().item(), 12)

    def test_modes_generate_the_same_question_stream_with_exact_budget(self):
        trainers = [Sampler("ordinary"), Sampler("dynamic")]
        update_counts = []
        for trainer in trainers:
            updates = 0
            while not trainer.sampling_stats.rollout_budget_exhausted:
                result = trainer._generate_and_score_completions([])
                self.assertEqual(result["advantages"].shape, (8,))
                updates += 1
            update_counts.append(updates)
            self.assertEqual(trainer.sampling_metrics["generated_groups"], 24)
        self.assertEqual(trainers[0].sampling_question_ids, trainers[1].sampling_question_ids)
        self.assertEqual(update_counts[0], 12)
        self.assertLess(update_counts[1], update_counts[0])

    def test_uniform_groups_use_bounded_fallback_and_exhaust_budget(self):
        trainer = Sampler("dynamic", budget=8, all_uniform=True, rounds=2)
        for _ in range(2):
            result = trainer._generate_and_score_completions([])
            self.assertEqual(result["advantages"].abs().sum().item(), 0)
        self.assertTrue(trainer.sampling_stats.rollout_budget_exhausted)
        self.assertEqual(trainer.sampling_metrics["fallback_zero_groups"], 4)
        with self.assertRaises(RuntimeError):
            trainer._generate_and_score_completions([])

    def test_refills_stop_at_budget_even_when_target_is_not_reached(self):
        trainer = Sampler("dynamic", budget=2, all_uniform=True, rounds=8)
        trainer._generate_and_score_completions([])
        self.assertEqual(len(trainer.sampling_question_ids), 2)
        self.assertTrue(trainer.sampling_stats.rollout_budget_exhausted)

    def test_unknown_rollout_fields_are_rejected(self):
        group = pack(2, 2, [0, 0])
        group["tool_mask"] = torch.ones(2, 2)
        with self.assertRaises(ValueError):
            merge_groups([group], pad_token_id=0)
