"""Stepwise code generation mode.

Provides stepwise code generation using multi-agent collaboration where specialized
agents handle different stages of the ML pipeline:
  - data_processing_and_feature_engineering
  - model_design
  - training_evaluation

Main entry: stepwise_plan_and_code_query()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from llm import generate, compile_prompt_to_md
from utils.response import extract_code, extract_text_up_to_code, wrap_code
from agents.prompts import is_optimization_or_rl_task
from agents.prompt_cache import dataset_reference_sentence, task_section

logger = logging.getLogger("MLEvolve")


def _generate_submission_enabled(agent_instance) -> bool:
    return getattr(agent_instance.acfg, "generate_submission", True)


def _optimization_rl_enabled(agent_instance, task_desc: str = "") -> bool:
    return is_optimization_or_rl_task(
        task_desc=task_desc or getattr(agent_instance, "task_desc", ""),
        coldstart_description=getattr(agent_instance, "coldstart_description", ""),
    )


@dataclass
class StepwiseContext:
    stage: str = "draft"
    memory: str = ""
    previous_code: str = ""
    execution_output: str = ""


@dataclass
class StepAgent:
    name: str
    introduction: str
    description: str
    guidelines: List[str]

    def generate(
        self,
        task_desc: str,
        data_preview: str,
        previous_steps: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        retries: int = 3,
        improvement_mode: bool = False,
        previous_module_code: str = "",
        improvement_strategy: str = "",
    ) -> Tuple[str, str]:
        prompt = self._build_prompt(
            task_desc=task_desc,
            data_preview_str=data_preview,
            previous_steps=previous_steps,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=context,
            improvement_mode=improvement_mode,
            previous_module_code=previous_module_code,
            improvement_strategy=improvement_strategy,
        )

        completion_text = None
        for _ in range(retries):
            completion_text = generate(
                prompt=prompt,
                temperature=agent_instance.acfg.code.temp,
                cfg=agent_instance.cfg
            )
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                return nl_text, code

            logger.debug(f"Extraction retry for {self.name}...")
        logger.warning(f"Code extraction failed after retries for {self.name}")
        return "", completion_text  # type: ignore

    def _build_prompt(
        self,
        task_desc: str,
        data_preview_str: str,
        previous_steps: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        improvement_mode: bool = False,
        previous_module_code: str = "",
        improvement_strategy: str = "",
    ) -> str | dict[str, str]:
        base_intro = prompt_base.get("Introduction", "")

        if context.stage == "improve":
            if improvement_mode and previous_module_code:
                step_specific_intro = (
                    f"You are currently working on improving the '{self.name}' step of the solution. "
                    f"Your task is to write ONLY the improved code for this specific step, based on the previous module code and the improvement strategy provided below. "
                    f"Improvement Strategy: {improvement_strategy if improvement_strategy else 'Improve this module based on the execution results.'}"
                )
            else:
                step_specific_intro = (
                    f"You are currently working on the '{self.name}' step of the solution. "
                    f"Your task is to write ONLY the code for this specific step that aligns with the overall improvement strategy. "
                    f"Base your implementation on the previous solution and execution results provided below, ensuring it integrates well with the improved approach."
                )
        else:
            step_specific_intro = (
                f"You are currently focusing on the '{self.name}' step of the solution. "
                f"Your task is to write ONLY the code for this specific step, not the complete solution."
            )
        introduction = base_intro + "\n\n" + step_specific_intro

        prev_summary = ""
        if previous_steps:
            prev_parts = []
            for step in previous_steps:
                prev_parts.append(f"### {step['name']}\n**Plan:** {step['plan']}\n**Code:**\n{wrap_code(step['code'])}")
            prev_summary = "\n\n".join(prev_parts)
        else:
            prev_summary = "This is the first step, no previous steps."

        generate_submission = _generate_submission_enabled(agent_instance)
        guidelines_to_use = self.guidelines.copy()
        current_step_description = self.description

        if self.name == "training_evaluation" and not generate_submission:
            current_step_description = (
                "Implement the training loop, validation, metric tracking, model saving, "
                "and configured non-submission artifacts."
            )
            guidelines_to_use.append(
                "CONFIG: Final submission generation is disabled. Do NOT force creation of `submission.csv`; "
                "focus on training, validation metric computation, and reusable inference code."
            )

        use_exact_coldstart_template = (
            hasattr(agent_instance, 'use_coldstart') and
            agent_instance.use_coldstart and
            hasattr(agent_instance, 'coldstart_description') and
            agent_instance.coldstart_description != "None model" and
            "Reference pattern" not in str(agent_instance.coldstart_description)
        )

        if use_exact_coldstart_template and context.stage == "draft":
            if self.name == "model_design":
                pretrain_emphasis = [
                    "**CRITICAL: You MUST prioritize using the recommended pretrained models provided in the Implementation guideline section below.**",
                    "The pretrained models are STRONGLY RECOMMENDED and should be your default first choice.",
                    "Only use custom architectures if the pretrained models are clearly unsuitable for this specific task."
                ]
                guidelines_to_use = pretrain_emphasis + guidelines_to_use
            elif self.name == "data_processing_and_feature_engineering":
                pretrain_awareness = [
                    "**IMPORTANT: Be aware that pretrained models may be used in later steps. Consider the input requirements of common pretrained models (e.g., image size, normalization, data format) when preparing the data and engineering features.**",
                    "For image tasks, ensure data is prepared in a format compatible with standard pretrained models (e.g., PIL Image, numpy arrays, proper image sizes).",
                    "For text tasks, ensure text data is properly tokenized and formatted for potential transformer models.",
                ]
                guidelines_to_use = pretrain_awareness + guidelines_to_use

        guidelines_text = "\n".join([f"- {g}" for g in guidelines_to_use])

        prompt_instructions = prompt_base["Instructions"].copy()

        prompt_instructions["Response format"] = (
            "Your response should be:\n"
            "1. A brief plan (2-3 sentences) describing what you will do in this step\n"
            "2. A single markdown code block (wrapped in ```) containing ONLY the code for this step\n"
            "IMPORTANT: Do NOT write code for other steps. Only write code for the current step."
        )

        prompt_instructions[f"{self.name} guidelines"] = [guidelines_text]

        if "Implementation guideline" in prompt_instructions:
            base_impl_guideline = prompt_instructions["Implementation guideline"]
            step_specific_impl = [
                "The code for this step must be self-contained and can be integrated with other steps.",
                "Use clear variable names that are consistent with previous steps.",
                "Do not duplicate code from previous steps - assume those parts already exist.",
                "Make sure to handle edge cases appropriately.",
            ]
            if isinstance(base_impl_guideline, list):
                prompt_instructions["Implementation guideline"] = base_impl_guideline + step_specific_impl
            else:
                prompt_instructions["Implementation guideline"] = [base_impl_guideline] + step_specific_impl

        prompt: Dict[str, Any] = {
            "Introduction": introduction,
            "Task description": task_desc,
            "Data preview": data_preview_str,
            "Memory": prompt_base.get("Memory", context.memory if context.memory else ""),
            "Previous steps": prev_summary,
            "Current step": {
                "Name": self.name,
                "Description": current_step_description,
            },
            "Instructions": prompt_instructions,
        }

        if context.stage == "improve":
            if improvement_mode and previous_module_code:
                prompt["Previous solution"] = {
                    "Code": wrap_code(previous_module_code),
                    "Note": f"This is the previous code for the '{self.name}' module. Improve it based on the improvement strategy provided above."
                }
            elif "Previous solution" in prompt_base:
                prompt["Previous solution"] = prompt_base["Previous solution"]
            elif context.previous_code:
                prompt["Previous solution"] = {
                    "Code": wrap_code(context.previous_code),
                }

        instructions = f"\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        if context.stage == "draft":
            okay_text = "Let me approach this systematically."
            assistant_suffix = ""
        elif context.stage == "improve":
            okay_text = "Let me approach this systematically."
            if improvement_mode and previous_module_code:
                previous_module_code_wrapped = wrap_code(previous_module_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously implemented the '{self.name}' module with the following code:\n{previous_module_code_wrapped}\n"
                    f"The execution of the full solution yielded the following results:\n{execution_output_wrapped}\n"
                    f"Improvement Strategy: {improvement_strategy if improvement_strategy else 'Improve this module based on the execution results.'}\n"
                    f"I need to improve this specific module according to the strategy above, ensuring it integrates well with the other modules."
                )
            elif context.previous_code:
                previous_code_wrapped = wrap_code(context.previous_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously made attempts with the following code:\n{previous_code_wrapped}\n"
                    f"The execution of this code yielded the following results:\n{execution_output_wrapped}\n"
                    f"I believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
                )
            else:
                assistant_suffix = ""
        else:
            okay_text = "Let me approach this systematically."
            assistant_suffix = ""

        model_name = agent_instance.acfg.code.model.lower()

        memory_section = ""
        if prompt.get("Memory", "").strip():
            if context.stage == "improve":
                memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"
            else:
                memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"

        previous_solution_section = ""
        if context.stage == "improve" and "Previous solution" in prompt:
            previous_solution_section = f"\n# Previous solution\n{prompt['Previous solution']['Code']}\n"

        user_prompt = (
            f"{task_section(prompt['Task description'], prompt['Data preview'])}\n"
            f"{instructions}"
            f"{memory_section}\n"
            f"{previous_solution_section}"
            f"# Previous steps\n{prompt['Previous steps']}\n\n"
            f"# Current step: {prompt['Current step']['Name']}\n{prompt['Current step']['Description']}\n\n"
        )
        assistant = f"{okay_text}\n{dataset_reference_sentence(prompt['Task description'], prompt['Data preview'])}{assistant_suffix}"
        return {
            "system": introduction,
            "user": user_prompt,
            "assistant": assistant,
        }



@dataclass
class MetaAgent:
    def merge(
        self,
        task_desc: str,
        data_preview_str: str,
        step_results: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        retries: int = 3,
    ) -> Tuple[str, str]:
        prompt = self._build_merge_prompt(
            task_desc=task_desc,
            data_preview_str=data_preview_str,
            step_results=step_results,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=context,
        )

        completion_text = None
        for _ in range(retries):
            completion_text = generate(
                prompt=prompt,
                temperature=agent_instance.acfg.code.temp,
                cfg=agent_instance.cfg
            )
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                return nl_text, code

            logger.debug("Extraction retry for MetaAgent merge...")
        logger.warning("Code extraction failed after retries for MetaAgent merge")
        return "", completion_text 

    def _build_merge_prompt(
        self,
        task_desc: str,
        data_preview_str: str,
        step_results: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        ) -> str | dict[str, str]:
        introduction = (
            "You are a Kaggle grandmaster attending a competition, an expert in writing clean, efficient, and competition-winning Python code for ML tasks. "
            "You have received code snippets from a team of specialized agents, each focusing on a specific part of the ML pipeline. "
            "Your critical task is to intelligently merge these partial scripts into a single, cohesive, and fully runnable Python script."
        )

        steps_summary = []
        for i, result in enumerate(step_results, 1):
            steps_summary.append(f"""
        ### Step {i}: {result['name']}
        **Plan:** {result['plan']}
        **Code:**
        {wrap_code(result['code'])}
        """)

        prompt_instructions = prompt_base["Instructions"].copy()

        prompt_instructions["Response format"] = (
            "Your response should be a brief summary (2-3 sentences) of how you merged the steps, "
            "followed by a single markdown code block (wrapped in ```) containing the complete merged code. "
            "There should be no additional headings or text in your response."
        )

        output_guideline = (
            "- Make sure the final code saves a reusable model artifact, defines `predict(model_path, data)`, prints validation metric (must match task's Evaluation section), and saves submission.csv"
            if _generate_submission_enabled(agent_instance)
            else "- Make sure the final code saves a reusable model artifact, defines `predict(model_path, data)`, and prints validation metric (must match task's Evaluation section); do not force submission.csv because final submission generation is disabled"
        )
        execution_flow = (
            "data processing & feature engineering -> RL environment / optimization solver design -> model design -> training & evaluation"
            if _optimization_rl_enabled(agent_instance, task_desc)
            else "data processing & feature engineering -> model design -> training & evaluation"
        )

        prompt_instructions["Merge guidelines"] = [
            "- Combine all code sections into a single, runnable Python script",
            "- CRITICAL: You are a MERGER, not a designer. Faithfully integrate the code from all steps. Do NOT introduce new models, algorithms, or approaches that were not in the original steps.",
            "- Ensure variable names are consistent across steps",
            "- Remove duplicate imports and definitions",
            "- Resolve conflicts between steps by following the earlier step's design (e.g., model_design defines the model, training_evaluation trains it)",
            f"- Ensure the execution flow is logical: {execution_flow}",
            output_guideline,
            "- The code should be a single-file Python program that can be executed as-is",
            "- The merged code MUST save the best trained model/preprocessing artifact under ./working, ./models, ./artifacts, or ./checkpoints.",
            "- The merged code MUST expose `def predict(model_path, data): ...` and validation/test/submission inference must use this function or the same internal inference routine.",
            "- Assume previous steps have NOT been executed; do not skip execution steps and only read files or outputs.",
            "- All parts must work together seamlessly",
        ]

        prompt: Dict[str, Any] = {
            "Introduction": introduction,
            "Task description": task_desc,
            "Memory": prompt_base.get("Memory", context.memory if context.memory else ""),
            "Data preview": data_preview_str,
            "Step results": "".join(steps_summary),
            "Instructions": prompt_instructions,
        }

        if context.stage == "improve":
            if "Previous solution" in prompt_base:
                prompt["Previous solution"] = prompt_base["Previous solution"]
            elif context.previous_code:
                prompt["Previous solution"] = {
                    "Code": wrap_code(context.previous_code),
                }

        instructions = f"\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        memory_section = ""
        if prompt.get("Memory", "").strip():
            if context.stage == "improve":
                memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"
            else:
                memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"

        okay_text = "Let me approach this systematically."

        if context.stage == "improve":
            if context.previous_code:
                previous_code_wrapped = wrap_code(context.previous_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously made attempts with the following code:\n{previous_code_wrapped}\n"
                    f"The execution of this code yielded the following results:\n{execution_output_wrapped}\n"
                    f"I believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
                )
            else:
                assistant_suffix = ""
        else:
            memory_section = f"# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}"
            okay_text = "Let me approach this systematically."
            assistant_suffix = ""

        user_prompt = (
            f"{task_section(prompt['Task description'], prompt['Data preview'])}\n"
            f"{instructions}"
            f"{memory_section}\n\n"
            f"# Step results\n{prompt['Step results']}\n\n"
        )
        assistant = f"{okay_text}\n{dataset_reference_sentence(prompt['Task description'], prompt['Data preview'])}{assistant_suffix}"
        return {
            "system": introduction,
            "user": user_prompt,
            "assistant": assistant,
        }


    def _simple_concat(self, step_results: List[Dict[str, str]]) -> str:
        code_parts = []
        for result in step_results:
            code_parts.append(f"# Step: {result['name']}\n{result['code']}\n")
        return "\n".join(code_parts)


def create_default_step_agents(include_rl: bool = False) -> List[StepAgent]:
    step_agents = [
        StepAgent(
            name="data_processing_and_feature_engineering",
            introduction="You are a Data Preparation Specialist responsible for data loading, cleaning, and feature engineering.",
            description="Load data from `./input` directory, perform cleaning, feature engineering, and create train/validation/test splits.",
            guidelines=[
                "Your responsibility: Load data from `./input`, clean, create features (preprocessing, encoding, augmentation), and split dataset into train/validation/test.",
                "CRITICAL: This step MUST include BOTH data loading AND feature engineering. Do NOT only load the raw data. You must actively create, transform, and enhance features to improve model performance.",
                "IMPORTANT: Apply feature engineering techniques such as feature scaling, encoding, transformation, and data augmentation methods appropriate for the task. Explore and implement feature engineering strategies that can enhance the model's ability to learn from the data.",
                "CRITICAL: Do NOT build models, write training code, or perform evaluation. Focus ONLY on data preparation and feature engineering.",
            ],
        )
    ]

    if include_rl:
        step_agents.append(
            StepAgent(
                name="rl_environment_design",
                introduction="You are an Optimization and Reinforcement Learning Environment Specialist responsible for formalizing decision processes.",
                description="For optimization/RL tasks, define the solver interface and, if RL is appropriate, implement the environment: state, action, transition, reward, terminal conditions, and validation hooks.",
                guidelines=[
                    "First decide whether RL is actually appropriate. If the task is a static one-shot optimization problem, implement a deterministic optimizer/heuristic interface instead of forcing RL.",
                    "If RL is chosen, explicitly define observation/state using only information available at decision time; never include future labels or evaluation-only information.",
                    "Define the action space, including action masks, feasibility repair, or decomposition for composite actions such as assign/order/route/schedule decisions.",
                    "Implement transition logic as deterministic, testable functions that update the partial solution, capacities, time, inventory, budget, position, or other state variables.",
                    "Design reward to align with the official objective/metric, with constraint penalties and optional dense shaping that does not change the final objective.",
                    "Define terminal and truncation conditions: completed assignment, infeasible dead-end, horizon/time limit, or validation episode end.",
                    "When RL is used, expose a Gymnasium-like API: `reset(seed=None, options=None)` and `step(action)` returning `(obs, reward, terminated, truncated, info)`.",
                    "Always provide a baseline solver path (greedy, local search, CP-SAT/MILP/OR-Tools when available) so training/evaluation can compare RL against non-RL optimization.",
                    "CRITICAL: Do NOT write the final model training loop here. Only provide environment/solver primitives, reward/objective functions, and validation helpers for later steps.",
                ],
            )
        )

    step_agents.extend([
        StepAgent(
            name="model_design",
            introduction="You are a Model Architect responsible for designing the model architecture, loss function, and optimizer.",
            description="Design the model architecture (including pretrained models), and define the loss function and optimizer.",
            guidelines=[
                "Your responsibility: Design the model architecture or choose reference pretrained model, loss function, and optimizer based on the task and the features from previous steps.",
                "For optimization/RL tasks, if the previous step chose RL, design the policy/value/Q-network and algorithm family to match the action space; otherwise design the non-RL scorer/model used by the optimizer.",
                "CRITICAL: Do NOT write the training loop, data processing, or feature engineering code. Only define the model, criterion, and optimizer objects.",
                "IMPORTANT: Consider the task's evaluation metric (from the task description's Evaluation section) when designing the model. The model output format should be compatible with the required evaluation metric calculation.",
                "IMPORTANT: When designing custom model architectures, include appropriate regularization components (e.g., Dropout layers) to prevent overfitting.",
            ],
        ),
        StepAgent(
            name="training_evaluation",
            introduction="You are a Training and Evaluation Expert responsible for implementing training, validation, and configured output generation.",
            description="Implement the training loop, validation, metric tracking, model saving, and configured output artifacts.",
            guidelines=[
                "Your responsibility: Write the training loop that uses the data, features, model, loss function, and optimizer from previous steps. Include validation, metric tracking, save the best model. Then load the best model, calculate validation metric (must match task's Evaluation section), and generate output artifacts only as required by config and task description.",
                "CRITICAL: Save a reusable best-model artifact and any preprocessing state under ./working, ./models, ./artifacts, or ./checkpoints. The final best_solution must contain this artifact, not only solution.py.",
                "CRITICAL: Define `def predict(model_path, data): ...`. It must load the artifact from `model_path`, apply the same preprocessing as validation/test, and return task-required predictions or decisions without retraining.",
                "For optimization/RL tasks, train/evaluate the chosen solver or policy against the exact objective; report both score and constraint violation statistics when constraints exist.",
                "CRITICAL: Assume that all previous code steps have already been executed. You should start directly from the training step. Do NOT redefine or reload the data, features, model, loss function, optimizer, environment, or solver primitives. These components are already defined and available from the previous steps.",
                "CRITICAL: You MUST use the variables and objects defined in previous steps AS-IS. Do NOT replace, redesign, or substitute them with different approaches. Your ONLY job is to write the training/evaluation code for what was already defined - not to introduce new models or pipelines.",
                "IMPORTANT: Your code should assume the data preprocessing, feature engineering, environment/solver design, and model design steps have been completed. Simply use the existing variables without copying them.",
                "CRITICAL: Validation metric computation must use the same prediction method as test inference, using training data only as reference, to avoid data leakage and ensure the metric reflects true generalization performance.",
                "CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
                "CRITICAL: You MUST actively prevent overfitting. Do NOT only focus on validation set metrics, as this can easily cause the model to overfit. You can consider to use standard anti-overfitting techniques as default modeling strategies, including:",
                "  - Data augmentation (when applicable to the task)",
                "  - Early stopping (monitor validation metric and stop when it stops improving)",
                "  - Regularization (weight decay, L1/L2 regularization)",
                "  - Dropout (if using neural networks)",
                "  - Other appropriate regularization techniques for the specific model type",
                "CRITICAL: You MUST implement the exact evaluation metric as specified in the task description's 'Evaluation' section. Read the Evaluation section carefully and implement it precisely according to the exact formula, calculation steps, and aggregation method described.",
                "CRITICAL: You MUST NOT use dummy, simplified, or approximate metrics. The validation metric must be a REAL and COMPLETE implementation of the task's evaluation metric as described in the Evaluation section, not an approximation, placeholder, or simplified version.",
                "CRITICAL: If the Evaluation section specifies multiple thresholds, components, or aggregation steps, you MUST implement ALL of them. Do not skip any required calculation steps or use shortcuts.",
                "CRITICAL: The metric calculation must match the Evaluation section exactly - use the same matching criteria, the same formula, the same thresholds (if any), and the same aggregation method as specified.",
                "CRITICAL: The final line must be: `print(f'Final Validation Score: {{score}}')`. This is required for the score parser.",
            ],
        ),
    ])
    return step_agents


def stepwise_plan_and_code_query(
    agent_instance,
    prompt_base: Dict[str, Any],
    data_preview: str,
    context: Dict[str, Any],
    ) -> Tuple[str, str]:
    logger.info("Using stepwise generation route.")

    stepwise_context = StepwiseContext(
        stage=context.get("stage", "draft"),
        memory=context.get("memory", ""),
        previous_code=context.get("previous_code", ""),
        execution_output=context.get("execution_output", ""),
    )

    step_agents = create_default_step_agents(
        include_rl=_optimization_rl_enabled(agent_instance, prompt_base["Task description"])
    )
    meta_agent = MetaAgent()

    step_results: List[Dict[str, str]] = []
    for idx, agent in enumerate(step_agents, 1):
        logger.info(f"Step {idx}/{len(step_agents)}: {agent.name}")

        plan, code = agent.generate(
            task_desc=prompt_base["Task description"],
            data_preview=data_preview,
            previous_steps=step_results,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=stepwise_context,
        )

        step_results.append({
            "name": agent.name,
            "plan": plan,
            "code": code,
        })

    logger.info("Merging all steps...")
    final_plan, final_code = meta_agent.merge(
        task_desc=prompt_base["Task description"],
        data_preview_str=data_preview,
        step_results=step_results,
        prompt_base=prompt_base,
        agent_instance=agent_instance,
        context=stepwise_context,
    )

    logger.info("Stepwise generation completed.")

    return final_plan, final_code
