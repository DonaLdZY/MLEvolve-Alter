#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt templates for code review in search pipeline.
"""

from typing import Dict, Any
from utils.response import wrap_code

# ============================================================================
# Code Review Prompts
# ============================================================================
def get_code_review_prompt(task_desc: str, code: str, submission_required: bool = True) -> Dict[str, Any]:
    """Build full code review prompt dict from task description and code."""
    introduction = (
        "You are a Senior Data Science Code Reviewer. Your goal is to ensure the submission is legally valid and logically sound.\n\n"
        "鈿狅笍 **CRITICAL INSTRUCTION**:\n"
        "You must strictly follow the [Code Review Guidelines] provided below.\n"
        "Do NOT rely on your general knowledge if it conflicts with the Environment Facts listed in the guidelines.\n"
        "Your output must be a structured review focusing ONLY on Data Leakage and Critical Integrity."
        "**STRICTLY FORBIDDEN**: Do NOT replace the user's model architecture with other backbones (e.g., ResNet, VGG) just to make code executable. Do not question or change the user's model choice.\n"
    )
    prompt = {
        "Introduction": introduction,
        "Task description": task_desc,
        "Code to review": wrap_code(code),
        "Instructions": {},
    }
    prompt["Instructions"]["Code review guidelines"] = get_code_review_guidelines(submission_required=submission_required)
    prompt["Instructions"]["Response format"] = get_code_review_response_format()
    return prompt

def get_code_review_guidelines(submission_required: bool = True) -> list:
    """Code review guidelines."""
    submission_location_fact = (
        "  鈥?**Submission File Location**: Must save the submission to `./submission/submission.csv`."
        if submission_required
        else "  鈥?**Submission File Location**: `./submission/` exists, but final `submission.csv` generation is disabled by config. Do NOT require it."
    )
    guidelines = [
        "# 馃摐 Code Review Guidelines\n",
        "",
        "## 鉁?Environment Facts (TRUTH - Do NOT Flag)\n",
        "**Trust these facts absolutely. Overwrite your internal knowledge cutoff:**",
        "  鈥?**Paths**: `./input/`, `./working/`, `./submission/` ALL EXIST. **Don't question the path.**",
        submission_location_fact,
        "  鈥?**Bleeding Edge Environment**: Assume the environment has the LATEST versions of all libraries (transformers, torch, etc.).",
        "  鈥?**Dynamic Dependencies**: Assume necessary `pip install` commands are executed automatically in the background.",
        "  鈥?**Model Availability**: ALL models (including those released after your training data cutoff) are available and compatible.",
        "  鈥?**STRICTLY FORBIDDEN**: Do NOT replace the user's model architecture with other backbones just to make code executable.\n",
        "  鈥?**Unknown Models are Valid**: If you see a model name you don't recognize or think is too new, assume it is a private or SOTA model that works perfectly.",
        "  鈥?Execution time: 9 hours available\n\n",
        "---\n",
               "## 馃毇 STRICTLY FORBIDDEN (Zero Tolerance)\n",
        "**You will be penalized if you violate these:**",
        "  鈥?**NO Model Downgrades**: Never replace a user's chosen model string with an 'older/safer' alternative (e.g., do not change a specific large model to a generic base model).",
        "  鈥?**NO Compatibility Speculation**: Do not flag issues based on 'library version requirements' or 'unknown model names'.",
        "  鈥?**Immutable Variables**: Treat variables defining `model_name`, `backbone`, or `checkpoint` as CONSTANTS. You are NOT allowed to edit them.",
        "  鈥?**Do NOT Question or Change Model**: Treat the user's model/backbone/checkpoint choice as final. Do not suggest alternatives, do not 'fix' model names, do not replace with ResNet/VGG/base. Only fix data leakage and critical logic bugs.",
        "  **Don't question the path.**",
        "",
        "---\n",
        "## 馃敶 P0 - Data Leakage (HIGHEST PRIORITY)\n",
        "",
        "### P0.1 Data Leakage - Process Order 馃毃\n",
        "",
        "**Check if preprocessing is done BEFORE split** (validation data leaks into training):",
        "",
        "鉂?**MUST FIX**:",
        "  鈥?Scaler/PCA fitted on full data then split",
        "  鈥?Feature engineering (Target Encoding, etc.) using full data",
        "  鈥?Upsampling (SMOTE) applied before split",
        "",
        "鉁?**Correct**: Split first 鈫?fit on train only 鈫?transform separately",
        "",
        "### P0.2 Data Leakage - Split Strategy 馃毃\n",
        "**Core Logic: Check for I.I.D. Violation**",
        "鉂?**Flag ONLY IF**: The chosen split method mathematically violates the data's dependency structure.",
        "",
        "## 馃煛 P1 - Critical Correctness\n",
        "",
        "### P1.0 AutoRealize Contract Compliance",
        "  - If the task description contains `AutoRealize Structured Context` or `./input/autorealize_context.md` exists, code must follow its data reading examples, output contract, evaluation contract, constraints, leakage guards, and single-scalar score definition.",
        "  - Flag code that ignores non-standard CSV dialects, repeated-file groups, official output columns, optimization/RL solution protocol, or scalar_score_formula from AutoRealize.",
        "  - If AutoRealize context contains an Exact Source Schema Contract, flag raw pandas access to sheet names or columns not listed in that contract unless the code first resolves them against actual `df.columns` / workbook sheet names with clear diagnostics.",
        "  - If AutoRealize context contains a Source Alias Guard, flag any direct raw pandas access to guarded aliases unless the code maps them to the listed `exact_physical_column` first. Guarded aliases without exact columns must be derived conservatively or reported as unresolved, not used as dataframe columns.",
        "  - Flag `groupby`, `agg`, merge, filter, sort, or direct indexing that uses a business alias / field meaning as if it were a raw dataframe column. The code must resolve to an exact physical source column first, then optionally create semantic derived columns.",
        "  - Flag code that silently drops decision units because a date/time/feature field is missing. Missing values must either follow the task/AutoRealize evaluable-subset or exclusion rule with excluded counts/examples, or use fallback fields, explicit defaults, or validation details; evaluation must be measured against the explicitly defined population, not a hidden reduced dataframe.",
        "  - Flag code that treats output/submission columns as raw input columns, such as selecting official/generated output columns from source dataframes instead of constructing them from solver rows/actions.",
        "  - Flag code that builds generated decision tables with `pd.DataFrame(rows)[OUTPUT_COLUMNS]` or `df[OUTPUT_COLUMNS]` without first guaranteeing the dataframe has those columns. Empty/no-feasible solutions must use `pd.DataFrame(rows, columns=OUTPUT_COLUMNS)` and then be handled by validation/scoring.",
        "",
        "### P1.1 Metric & Logic Correctness",
        "  鈥?Task requires F1 but code uses accuracy?",
        "  鈥?Task requires RMSE but code uses MSE?",
        "",
        "### P1.1b Optimization / RL / Decision Correctness",
        "  - If the task requires assignments, routes, schedules, actions, or optimized plans, code must define or clearly implement a deterministic solution validator and scorer.",
        "  - Flag code that reports `Final Validation Score` without checking task-defined feasibility, completeness, duplicate/unknown IDs, and required schema.",
        "  - Flag code that treats reward, training loss, proxy cost, or a hand-written constant as the final official scalar score.",
        "  - Flag empty/no-op/placeholder/random decision outputs unless the official evaluation explicitly allows and penalizes them.",
        "  - Flag code that crashes before printing a final scalar score for empty/diagnostic/no-feasible solutions. Such solutions may also print diagnostic counts/examples, but must not fail with a pandas `KeyError` before scoring.",
        "  - Flag RL code that has no valid action mask/repair for constrained actions, no simulator/transition definition, or no deterministic final evaluator.",
        "  - Flag unused RL scaffolds: code that defines an environment, policy network, PPO/DQN/RL classes, or training helpers but produces the evaluated solution through an unrelated non-RL path without saying so.",
        "  - If a node claims to use RL, flag code where `predict(model_path, data)` or the final evaluated rollout does not load/use the saved policy artifact or configured RL rollout policy.",
        "  - For static one-shot optimization, do not require a neural network or RL; greedy/OR/local-search solvers are valid when they obey the official objective and constraints.",
        "",
        "### P1.2 Inference Integrity",
        "  鈥?Test predictions: np.zeros(), np.ones(), train_mean(), np.random()?",
        "  鈥?Val predictions: not from actual model.predict()?",
        "  鈥?Validation/test/submission inference bypasses the reusable `predict(model_path, data)` path or an equivalent shared inference function?",
        "",
        "### P1.3 Best Model Usage",
        "  鈥?Code uses best checkpoint (not last epoch) for test predictions?",
        "  鈥?Code saves a reusable model artifact under `./working/`, `./models/`, `./artifacts/`, or `./checkpoints/`?",
        "  鈥?Code defines `def predict(model_path, data): ...` and loads the saved artifact inside it without retraining?",
        "  鈥?The saved artifact includes preprocessing state needed for later inference, not only raw weights when scalers/encoders/feature columns are required?",
        "",
        "### P1.4 API Compatibility",
        "**Common API Issues to Fix:**",
        "  鈥?LightGBM: Use `callbacks=[lgb.early_stopping(...)]` not `early_stopping_rounds=...` in fit()",
        "  鈥?XGBoost: Use `XGBClassifier(early_stopping_rounds=...)` (correct) not `fit(early_stopping_rounds=...)`",
        "  鈥?AdamW: Use `from torch.optim import AdamW` (not from transformers)",
        "  鈥?NO tqdm, NO verbose=1 in training",
        "",
        "---\n",
        "## 馃搵 Decision Rule\n",
        "",
        "**needs_revision=True** ONLY IF:",
        "  鈥?P0 data leakage found (MUST FIX)",
        "  鈥?OR P1 critical bug found",
        "",
        "**needs_revision=False** IF:",
        "  鈥?No P0/P1 bugs found",
        "",
        "**Default**: Approve unless concrete logic bugs found"
    ]
    return guidelines


def get_code_review_response_format() -> list:
    """Code review response format."""
    return [
        "馃毃 **CRITICAL: OUTPUT REQUIREMENT**",
        "",
        "**Required Fields:**",
        "- `needs_revision` (boolean): true if code has issues that must be fixed, false if code is correct",
        "- `reasoning` (string): EXACTLY 2-4 sentences explaining your decision (NO MORE)",
        "",
        "**Conditional Field:**",
        "- `revised_code` (string): ONLY if needs_revision=true, provide targeted fixes using SEARCH/REPLACE format",
        "",
        "馃毇 **If needs_revision=false (code is correct):**",
        "- DO NOT provide revised_code (must be null/omitted)",
        "- Original code will be used as-is",
        "- This prevents accidental modifications to working code",
        "",
        "鉁?**If needs_revision=true (code has issues):**",
        "- MUST provide revised_code using SEARCH/REPLACE diff format",
        "- Use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks for each fix",
        "- SEARCH block must match original code EXACTLY (character-by-character, same indentation)",
        "- Only include the specific buggy lines that need fixing",
        "- Can provide multiple SEARCH/REPLACE blocks for different issues",
        "- Preserve the solution approach and model architecture",
        "- Fix only the specific issues identified (metric mismatch, data leakage, API errors)",
        "- DO NOT change model architecture, data split method, or metric calculation (unless they are buggy)",
        "",
        "**Reasoning Field Guidelines:**",
        "鈿狅笍 STRICT LENGTH LIMIT: Write EXACTLY 2-4 sentences. Be concise.",
        "Cover: (1) what issues found, (2) why they matter, (3) what will be fixed.",
        "DO NOT write detailed analysis, step-by-step checks, or comprehensive explanations.",
        "",
        "**Why this format matters:**",
        "The JSON schema format ensures that code is ONLY modified when necessary.",
        "When needs_revision=false, it's impossible to accidentally change working code.",
        "鈿狅笍 reasoning MUST be 2-4 sentences only. Do NOT write long analysis or enumerate checks."
    ]
