"""Code Review Agent: LLM-based code review and fix for node code."""

import logging
import time
from typing import cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from agents.prompts.validation_template_prompts import get_code_review_prompt
from agents.prompts import get_internet_clarification
from agents.prompt_cache import task_section
from utils.autorealize_context import select_autorealize_context_for_stage

from agents.coder.diff_coder import SearchReplacePatcher

logger = logging.getLogger("MLEvolve")

CODE_REVIEW_SPEC = FunctionSpec(
    name="submit_code_review",
    json_schema={
        "type": "object",
        "properties": {
            "needs_revision": {
                "type": "boolean",
                "description": (
                    "true if the code has issues that must be fixed "
                    "(metric mismatch, data leakage, or missing packages), "
                    "false if the code is correct."
                )
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "CONCISE explanation in EXACTLY 2-4 sentences. Explain: "
                    "(1) what issues were found, (2) why they matter, (3) what will be fixed. "
                    "DO NOT write detailed analysis or step-by-step checks - keep it brief."
                )
            },
            "revised_code": {
                "type": "string",
                "description": (
                    "ONLY if needs_revision=true: Provide targeted fixes using SEARCH/REPLACE diff format.\n\n"
                    "**REQUIRED FORMAT** (use this for each fix):\n"
                    "<<<<<<< SEARCH\n"
                    "[exact code to find - copy verbatim with exact indentation]\n"
                    "=======\n"
                    "[corrected code]\n"
                    ">>>>>>> REPLACE\n\n"
                    "**CRITICAL**: \n"
                    "- SEARCH block must match original code EXACTLY (character-by-character, including all spaces/tabs)\n"
                    "- Only include the specific buggy lines that need fixing\n"
                    "- Can provide multiple SEARCH/REPLACE blocks for different bugs\n"
                    "- Do NOT output complete code - only diff blocks\n"
                    "- Do NOT wrap output in markdown code fences (``` or ```python) - output raw diff only\n\n"
                    "If needs_revision=false: MUST be null (DO NOT output code)."
                )
            }
        },
        "required": ["needs_revision", "reasoning"]
    },
    description="Submit code review for search node solution."
)


def run(agent, node: SearchNode) -> str:
    logger.debug(f"[review] node {node.id}")

    review_context = select_autorealize_context_for_stage(
        str(getattr(agent, "autorealize_context", "") or ""),
        "code_review",
    )
    review_task_desc = str(agent.task_desc or "")
    if review_context:
        review_task_desc = f"{review_task_desc}\n\n{review_context}".strip()
    prompt = get_code_review_prompt(
        task_desc=review_task_desc,
        code=node.code,
        submission_required=getattr(agent.acfg, "generate_submission", True),
    )
    internet_clarification = get_internet_clarification(getattr(agent.cfg, "pretrain_model_dir", ""))
    if "Instructions" not in prompt:
        prompt["Instructions"] = {}
    if "Implementation guideline" in prompt["Instructions"]:
        prompt["Instructions"]["Implementation guideline"].extend(internet_clarification)
    else:
        prompt["Instructions"]["⚠️ Internet Access Clarification"] = internet_clarification

    use_diff_for_review = agent.acfg.use_diff_mode
    retry_cfg = getattr(agent.acfg, "retries", None)
    max_retries = max(1, int(getattr(retry_cfg, "code_review_max_attempts", 2)))
    retry_delay = max(0.0, float(getattr(retry_cfg, "code_review_delay_seconds", 5.0)))
    primary_role = str(getattr(retry_cfg, "code_review_model_role", "feedback") or "feedback").strip().lower()
    if primary_role not in {"feedback", "code"}:
        logger.warning("Unknown code_review_model_role=%r; using feedback", primary_role)
        primary_role = "feedback"
    roles = [primary_role]
    if bool(getattr(retry_cfg, "code_review_escalate_to_code", True)) and primary_role != "code":
        roles.append("code")
    roles = roles[:max_retries]

    for attempt, role in enumerate(roles):
        try:
            if attempt > 0:
                logger.info(
                    "Escalating code review to role=%s (attempt %s/%s) for node %s",
                    role,
                    attempt + 1,
                    len(roles),
                    node.id,
                )
                time.sleep(retry_delay)

            stage_cfg = agent.acfg.feedback if role == "feedback" else agent.acfg.code

            review_response = cast(
                dict,
                query(
                    system_message={
                        "Introduction": prompt.get("Introduction", ""),
                        "Instructions": prompt.get("Instructions", {}),
                    },
                    user_message=(
                        f"{task_section(review_task_desc)}\n"
                        f"# Code to review\n{prompt.get('Code to review', '')}"
                    ),
                    func_spec=CODE_REVIEW_SPEC,
                    model=stage_cfg.model,
                    temperature=stage_cfg.temp,
                    stage_name=role,
                    cfg=agent.cfg
                ),
            )

            needs_revision = review_response.get("needs_revision", False)
            reasoning = review_response.get("reasoning", "")
            revised_code = review_response.get("revised_code")
            logger.info(
                "Code review for node %s using role=%s: needs_revision=%s",
                node.id,
                role,
                needs_revision,
            )
            logger.info(f"Reasoning: {reasoning}", extra={"verbose": True})

            if needs_revision:
                if revised_code and revised_code.strip():
                    if use_diff_for_review and (
                        "<<<<<<< SEARCH" in revised_code or "< SEARCH" in revised_code
                        ):
                        try:
                            logger.info("Code review returned diff format, applying patch")
                            patcher = SearchReplacePatcher()
                            patched_code, count = patcher.apply_patch(
                                revised_code, node.code, strict=False
                            )
                            if count > 0 and patched_code and patched_code != node.code:
                                logger.info(f"Successfully applied {count} review patch(es)")
                                return patched_code.strip()
                            raise ValueError(f"review diff did not apply (count={count})")
                        except Exception as e:
                            logger.warning(
                                "Failed to apply %s review patch: %s",
                                role,
                                e,
                            )
                            if attempt < len(roles) - 1:
                                continue
                            return node.code
                    else:
                        # Full code revision (original behavior)
                        if use_diff_for_review:
                            logger.warning(
                                "%s reviewer returned full code while diff mode is enabled.",
                                role,
                            )
                            if attempt < len(roles) - 1:
                                continue
                            return node.code
                        else:
                            logger.info("Using revised code from reviewer")
                            return revised_code.strip()

                if attempt < len(roles) - 1:
                    logger.warning(
                        "Code review requested a revision without an applicable patch; escalating."
                    )
                    logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                    continue
                logger.error("Code review requested a revision without an applicable patch; returning original code")
                logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                return node.code

            if revised_code is not None and revised_code.strip():
                logger.warning(
                    "Code review warning: needs_revision=False but revised_code was provided. "
                    "Ignoring revised_code and using original code."
                )
            logger.info("Code approved, using original code")
            return node.code

        except Exception as e:
            error_msg = f"Code review failed with exception: {e}"
            if attempt < len(roles) - 1:
                logger.warning("%s - escalating to the next review role", error_msg)
                continue
            logger.error(f"{error_msg} - returning original code")
            return node.code

    logger.error("Code review: Unexpected exit from retry loop, returning original code")
    return node.code
