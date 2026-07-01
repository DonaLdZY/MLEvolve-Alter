"""Implementation guideline."""

import time

import humanize

from .shared import (
    get_decision_solution_protocol,
    get_optimization_rl_strategy,
    is_optimization_or_rl_task,
)


def get_impl_guideline_from_agent(agent):
    """Build implementation guideline from agent config."""
    tot_time_remaining = agent.acfg.time_limit - (time.time() - agent.start_time)
    exec_timeout = int(min(agent.cfg.exec.timeout, tot_time_remaining))
    optimization_rl = is_optimization_or_rl_task(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
    )
    guideline = get_impl_guideline(
        tot_time_remaining=tot_time_remaining,
        steps_remaining=agent.acfg.steps - agent.current_step,
        exec_timeout=exec_timeout,
        expose_prediction=getattr(agent.acfg, "expose_prediction", False),
        k_fold_validation=getattr(agent.acfg, "k_fold_validation", 0),
        pretrain_model_dir=getattr(agent.cfg, "pretrain_model_dir", ""),
        generate_submission=getattr(agent.acfg, "generate_submission", True),
        optimization_rl=optimization_rl,
    )
    guideline |= get_decision_solution_protocol(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
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
    optimization_rl: bool = False,
) -> dict:
    """Build implementation guideline from time and config."""
    prediction_scope = (
        "validation & test"
        if generate_submission
        else "validation and any explicitly requested downstream inference output"
    )
    if optimization_rl:
        inference_guideline = [
            "**1. Decision / Solution Generation for Optimization or RL**",
            "- EVERY reported decision, assignment, route, schedule, action, or output row must come from a real solver, policy, heuristic, or optimization procedure in the code.",
            "- Required functions: `load_problem_data(input_dir)`, `build_solution(data)`, `validate_solution(solution, data)`, `score_solution(solution, data)`, and `run_evaluator_self_tests(data)`.",
            "- If AutoRealize provides an Exact Source Schema Contract, use only exact physical sheet/column names for raw dataframe access. Business concepts and English variable names must be created after loading through an explicit mapping.",
            "- Implement a safe column resolver for business aliases: exact match first, then conservative alias/fuzzy matching against actual columns with diagnostics. Never access a column name that is absent from `df.columns`.",
            "- Before `groupby`, `agg`, merge, filter, or sort, resolve every business concept to an exact physical source column variable and use that variable in the pandas call. Semantic names may be created only as derived/code-local columns after exact source access.",
            "- Define the evaluation population before scoring. Missing date/time/feature values must not be silently dropped. If the AutoRealize/task contract defines a valid/evaluable subset or exclusion rule, apply it explicitly and report excluded counts/examples; otherwise use fallback fields, explicit defaults, or validation notes.",
            "- `validate_solution` must check task-defined feasibility and completeness before any score is trusted.",
            "- `score_solution` must implement the single scalar score from the task description / AutoRealize evaluation contract. Do not report reward-only values, training loss, proxy costs, or extra metrics as `Final Validation Score`.",
            "- FORBIDDEN: empty/no-op/constant/placeholder/random solutions being treated as competitive unless the official invalid-output rule explicitly scores them and the validation summary says so.",
            "- Output/submission columns describe the generated result table, not raw input dataframe columns. Never select `output_columns` from a source dataframe, and never assume names like solution name, delivery day, trip id, validity, or failure reason exist in the input unless the Exact Source Schema Contract lists them exactly.",
            "- When writing a generated decision/result table, declare `OUTPUT_COLUMNS` and use `pd.DataFrame(rows, columns=OUTPUT_COLUMNS)`. This preserves schema for empty/no-feasible solutions and prevents `KeyError` from slicing a zero-column dataframe.",
            "- Empty or diagnostic solutions must still reach `validate_solution`, `score_solution`, and `Final Validation Score`; they should be penalized or marked infeasible according to the contract, not crash during output creation.",
            "",
        ]
        metric_guideline = [
            "**5. Print Validation Metric with Decision Evidence**",
            "- You may print one JSON diagnostic line before the final score: `Decision Validation Summary: {...}`.",
            "- For heuristic or rule-based solvers, `model_path` may be a null/ignored placeholder, but `predict(model_path, data)` must still exist as the reusable entrypoint.",
            "- If you print the JSON, include task-defined diagnostics that help later nodes improve; no specific diagnostic field is required for acceptance.",
            "- The JSON should also include task-relevant progress signals when meaningful, but do not invent universal progress or violation fields for tasks that do not define them.",
            "- If validation fails or the solution is partial/infeasible, include counts and examples using the task's own concepts: invalid actions, missing entities, duplicate IDs, violated constraints, infeasibility reasons, or other relevant diagnostics.",
            "- MUST print the final line exactly as: `print(f'Final Validation Score: {score}')`.",
            "- Score MUST come from `score_solution(solution, data)` after `validate_solution` and official invalid-solution handling.",
            "- If validation finds task-defined infeasibility, apply the official infeasible-solution penalty or worst score. Never let an infeasible solution look best.",
            "- Partial but real solutions may be retained for improvement when the official score penalizes remaining invalid/incomplete cases. Empty/no-op outputs must be reported as diagnostics, not claimed as competitive.",
            "- If output generation is enabled, save a schema-correct table even when there are zero generated rows: `pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(...)`.",
            "",
        ]
        predict_self_check = [
            "Did I generate a real feasible solution/action plan through a solver, policy, or heuristic rather than placeholders?",
            "Did `validate_solution` check task-defined feasibility, completeness, uniqueness, schema, and unknown IDs/entities?",
            "Did `score_solution` compute exactly one official scalar score and did `Final Validation Score` come from it?",
            "Did I print a final scalar score, with optional diagnostics when they help later improvement?",
            "Did I construct generated output tables with explicit columns and avoid treating output columns as raw input columns?",
        ]
    else:
        inference_guideline = [
            "**1. Model Inference for ALL Predictions**",
            f"- EVERY prediction ({prediction_scope}) MUST come from trained model's forward pass",
            "- Process: Load data -> Preprocess -> model.predict()/model.forward() -> Save predictions",
            "- FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers",
            "- FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)",
            "- Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)",
            "",
        ]
        metric_guideline = [
            "**5. Print Validation Metric**",
            "- MUST print: `print(f'Final Validation Score: {score}')`",
            "- Score MUST be computed on hold-out validation set using proper metric formula",
            "- CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
            "",
        ]
        predict_self_check = [
            "Did predictions pass through model's learned weights during inference? (If NO -> INVALID)",
        ]

    submission_guideline = (
        [
            "**2. Generate submission.csv**",
            "鈥?Path: `./submission/submission.csv` (NOT ./working/submission.csv)",
            "鈥?Content: Model predictions or decision/solution rows for ALL required evaluation units",
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
        "- If AutoRealize context contains `Source Alias Guard`, every listed alias is unsafe for raw pandas access unless it gives `exact_physical_column`; do not use guarded aliases in `df[...]`, `groupby`, `merge(on=...)`, or `sheet_name`.",
        "",
        *inference_guideline,
        *submission_guideline,
        "**3. Save Reusable Model Artifact**",
        "鈥?MUST save the trained best model and all required preprocessing state to disk under `./working/`, `./models/`, `./artifacts/`, or `./checkpoints/`.",
        "鈥?Use a standard artifact filename such as `./working/model_artifact.pkl`, `./working/best_model.pt`, or `./working/best_model.joblib`; the executor may rewrite generic filenames per node to avoid conflicts.",
        "鈥?The artifact must be sufficient for later inference without retraining: include fitted scalers/encoders/tokenizers/label maps/feature columns and model weights, solver state, policy checkpoint, heuristic parameters, or solver configuration.",
        "鈥?For PyTorch, save a checkpoint dict containing `state_dict` plus preprocessing/config metadata when needed. For sklearn/XGBoost/LightGBM/CatBoost, save the model pipeline or a dict with model plus preprocessing objects using joblib/pickle/native save.",
        "",
        "**4. Expose Reusable Inference API**",
        "鈥?MUST define `def predict(model_path, data): ...` in the final script.",
        "鈥?`predict(model_path, data)` must load the saved artifact from `model_path`, apply the same preprocessing as validation/test, and return raw predictions, decisions, or the task-required solution object without retraining.",
        "鈥?Validation inference, test inference, and submission generation must use this function or exactly the same internal inference routine. Do NOT retrain inside `predict`.",
        "",
        *metric_guideline,
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
        *[f"鈻?{item}" for item in predict_self_check],
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
