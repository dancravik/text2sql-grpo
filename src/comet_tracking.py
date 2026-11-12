"""Start or reopen a Comet experiment for training and SQL evaluation."""

import os
from pathlib import Path


def start_experiment(*, name: str | None = None, key: str | None = None):
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    api_key = os.environ.get("COMET_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set COMET_API_KEY in the environment or in the project .env")

    import comet_ml

    config = comet_ml.ExperimentConfig(
        name=name,
        log_code=False,
        log_git_patch=False,
        log_env_details=False,
        auto_output_logging=False,
        auto_param_logging=False,
        auto_metric_logging=False,
    )
    if key:
        return comet_ml.start(api_key=api_key, experiment_key=key, mode="get", experiment_config=config)
    return comet_ml.start(
        api_key=api_key,
        workspace=os.environ.get("COMET_WORKSPACE") or None,
        project_name=os.environ.get("COMET_PROJECT_NAME") or "text2sql-grpo",
        mode="create",
        experiment_config=config,
    )
