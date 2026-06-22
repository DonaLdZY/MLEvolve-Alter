"""Implementation guideline."""

import time

import humanize

from .shared import get_optimization_rl_strategy


def get_impl_guideline_from_agent(agent):
    """Build implementation guideline from agent config."""
    tot_time_remaining = agent.acfg.time_limit - (time.time() - agent.start_time)
    exec_timeout = int(min(agent.cfg.exec.timeout, tot_time_remaining))
    guideline = get_impl_guideline(
        tot_time_remaining=tot_time_remaining,
        steps_remaining=agent.acfg.steps - agent.current_step,
        exec_timeout=exec_timeout,
        expose_prediction=getattr(agent.acfg, "expose_prediction", False),
        k_fold_validation=getattr(agent.acfg, "k_fold_validation", 0),
        pretrain_model_dir=getattr(agent.cfg, "pretrain_model_dir", ""),
        generate_submission=getattr(agent.acfg, "generate_submission", True),
    )
    guideline |= get_optimization_rl_strategy(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
    )
    return guideline


def _format_time(time_in_sec):
    """Format seconds for display."""
    return f"{int(time_in_sec) // 3600}h {(int(time_in_sec) % 3600) // 60}m {int(time_in_sec) % 60}s"


def get_impl_guideline(
    tot_time_remaining: float,
    steps_remaining: int,
    exec_timeout: int,
    expose_prediction: bool = False,
    k_fold_validation: int = 0,
    pretrain_model_dir: str = "",
    generate_submission: bool = True,
) -> dict:
    """Build implementation guideline from time and config."""
    prediction_scope = (
        "validation & test"
        if generate_submission
        else "validation and any explicitly requested downstream inference output"
    )
    submission_guideline = (
        [
            "**2. Generate submission.csv**",
            "鈥?Path: `./submission/submission.csv` (NOT ./working/submission.csv)",
            "鈥?Content: Model predictions on ALL test samples",
            "鈥?Format: Follow task description exactly",
            "",
        ]
        if generate_submission
        else [
            "**2. Final submission generation is disabled by config**",
            "鈥?Do NOT force creation of `./submission/submission.csv`.",
            "鈥?Focus on a real train/validation pipeline, rigorous metric computation, and reusable inference code.",
            "鈥?Only create an output file if the task description explicitly requires a non-submission deliverable with a clear schema.",
            "",
        ]
    )
    directories_guideline = (
        "馃搧 **Directories**: Input data in `./input/`, submission in `./submission/`, temp files in `./working/`"
        if generate_submission
        else "馃搧 **Directories**: Input data in `./input/`, temp files in `./working/`; `./submission/` exists but final submission is not required by config"
    )
    submission_self_check = (
        ["鈻?Did I generate submission.csv in correct path with ALL test predictions?"]
        if generate_submission
        else ["鈻?Did I avoid forcing submission.csv because final submission generation is disabled?"]
    )
    impl_guideline = [
        f"**Resource Budget**: Time left 鈮?{_format_time(tot_time_remaining)} | Steps left = {steps_remaining} | Max execution time per run = {humanize.naturaldelta(exec_timeout)}",
        "",
        "**Note:** Code execution MUST complete within 9 hours (hard limit) 鈥?any solution exceeding this will be invalid. Within this constraint, prioritize performance and optimization.",
        "馃幆 **CRITICAL REQUIREMENTS** (Non-Negotiable):",
        "",
        "**0. AutoRealize Contract Priority**",
        "- If `./input/autorealize_context.md` exists, read and obey it before using generic assumptions.",
        "- Its data reading examples, output contract, evaluation contract, constraints, leakage guards, and single-scalar score definition override generic Kaggle templates and lightweight file previews.",
        "- Do not invent submission columns, target fields, row-count rules, random seeds, distance matrices, or cost formulas when AutoRealize did not provide authority for them.",
        "",
        "**1. Model Inference for ALL Predictions**",
        f"鈥?EVERY prediction ({prediction_scope}) MUST come from trained model's forward pass",
        "鈥?Process: Load data 鈫?Preprocess 鈫?model.predict()/model.forward() 鈫?Save predictions",
        "鈥?鉂?FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers",
        "鈥?鉂?FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)",
        "鈥?Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)",
        "",
        *submission_guideline,
        "**3. Save Reusable Model Artifact**",
        "鈥?MUST save the trained best model and all required preprocessing state to disk under `./working/`, `./models/`, `./artifacts/`, or `./checkpoints/`.",
        "鈥?Use a standard artifact filename such as `./working/model_artifact.pkl`, `./working/best_model.pt`, or `./working/best_model.joblib`; the executor may rewrite generic filenames per node to avoid conflicts.",
        "鈥?The artifact must be sufficient for later inference without retraining: include fitted scalers/encoders/tokenizers/label maps/feature columns and model weights or solver state.",
        "鈥?For PyTorch, save a checkpoint dict containing `state_dict` plus preprocessing/config metadata when needed. For sklearn/XGBoost/LightGBM/CatBoost, save the model pipeline or a dict with model plus preprocessing objects using joblib/pickle/native save.",
        "",
        "**4. Expose Reusable Inference API**",
        "鈥?MUST define `def predict(model_path, data): ...` in the final script.",
        "鈥?`predict(model_path, data)` must load the saved artifact from `model_path`, apply the same preprocessing as validation/test, and return raw predictions or task-required decision outputs.",
        "鈥?Validation inference, test inference, and submission generation must use this function or exactly the same internal inference routine. Do NOT retrain inside `predict`.",
        "",
        "**5. Print Validation Metric**",
        "鈥?MUST print: `print(f'Final Validation Score: {score}')`",
        "鈥?Score MUST be computed on hold-out validation set using proper metric formula",
        "鈥?CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
        "",
        directories_guideline,
        "",
        f"馃摝 **Packages & Internet**: numpy, pandas, sklearn, torch, transformers, timm, xgboost, lightgbm (all pre-installed). torch.hub.load(), HuggingFace, etc. available during development."
        + (f" Offline models at `{pretrain_model_dir}`" if pretrain_model_dir else ""),
        "",
        "鈿狅笍 **API Compatibility**: LightGBM/XGBoost: 鉂?`fit(..., early_stopping_rounds=...)` 鈫?鉁?LightGBM: `fit(..., callbacks=[lgb.early_stopping(...)])` 鉁?XGBoost: `XGBClassifier(early_stopping_rounds=...)`",
        "鈥?AdamW: 鉂?`from transformers import AdamW` (deprecated) 鈫?鉁?`from torch.optim import AdamW`",
        "",
        "馃毇 **Execution Guidelines**:",
        "鈥?NO tqdm (not installed), NO verbose=1",
        "鈥?Print only 1 line per epoch (minimize logging)",
        "鈥?Use DataLoader with num_workers>=2 for speed",
        "",
        "鈿狅笍  **Self-Check Before Finalizing**:",
        "鈻?Did predictions pass through model's learned weights during inference? (If NO 鈫?INVALID)",
        "鈻?Did I save the best trained model/preprocessing artifact under ./working, ./models, ./artifacts, or ./checkpoints?",
        "鈻?Did I define `predict(model_path, data)` and use it or the same inference path for validation/test/submission?",
        *submission_self_check,
        "鈻?Did I print validation metric as the last line?",
        "鈻?Did I use the COMPLETE training dataset (not a tiny subset)?",
    ]
    if expose_prediction:
        impl_guideline.append(
            "Because prediction exposure is enabled, document the expected `data` input type for `predict(model_path, data)` and keep the function usable from another Python process."
        )

    if k_fold_validation > 1:
        impl_guideline.append(
            f"The evaluation should be based on {k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
        )

    return {"Implementation guideline": impl_guideline}
