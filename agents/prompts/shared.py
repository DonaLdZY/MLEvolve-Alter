"""Static shared prompt fragments."""

from __future__ import annotations


OPTIMIZATION_RL_KEYWORDS = (
    "reinforcement learning",
    "offline rl",
    "online rl",
    "mdp",
    "markov decision",
    "policy learning",
    "reward function",
    "gymnasium",
    "gym env",
    "environment step",
    "simulator",
    "sequential decision",
    "dynamic decision",
    "routing",
    "vehicle routing",
    "scheduling",
    "assignment problem",
    "resource allocation",
    "portfolio optimization",
    "knapsack",
    "combinatorial optimization",
    "optimization problem",
    "constraint solver",
    "cp-sat",
    "mixed integer",
    "integer programming",
    "linear programming",
    "强化学习",
    "离线强化学习",
    "在线强化学习",
    "马尔可夫决策",
    "状态空间",
    "动作空间",
    "奖励函数",
    "策略学习",
    "仿真环境",
    "序列决策",
    "路径规划",
    "路径优化",
    "车辆路径",
    "调度",
    "排程",
    "分配问题",
    "资源分配",
    "组合优化",
    "运筹优化",
    "整数规划",
)


def is_optimization_or_rl_task(task_desc: str = "", coldstart_description: str = "") -> bool:
    """Return True only when the task text strongly suggests optimization/RL."""
    text = f"{task_desc}\n{coldstart_description}".lower()
    if not text.strip():
        return False
    if "Model" in coldstart_description and "Optimization" in coldstart_description:
        return True
    return any(keyword in text for keyword in OPTIMIZATION_RL_KEYWORDS)


def get_optimization_rl_strategy(task_desc: str = "", coldstart_description: str = "") -> dict:
    """Conditional guidance for optimization and reinforcement-learning tasks."""
    if not is_optimization_or_rl_task(task_desc, coldstart_description):
        return {}

    return {
        "Optimization / Reinforcement Learning Strategy (conditional)": [
            "",
            "**Activate this section only because the task appears to involve optimization, sequential decision-making, or RL. Do not force RL if the problem is better solved by operations research or heuristics.**",
            "",
            "**First decide the problem family:**",
            "- Static one-shot assignment/routing/scheduling/resource allocation: start with greedy baselines, MILP/CP-SAT/OR-Tools if constraints are explicit, then local search, simulated annealing, tabu search, genetic/evolutionary search, or large-neighborhood search.",
            "- Sequential decision task with a simulator, transition dynamics, delayed rewards, or logged trajectories: consider an RL formulation.",
            "- If there are only historical decisions and no safe simulator for exploration, treat it as offline RL or imitation learning; do not use naive online exploration.",
            "",
            "**If proposing RL, the plan must explicitly define:**",
            "- State / observation: all information available at decision time only; no future leakage.",
            "- Action space: discrete, continuous, or composite actions; include feasibility masks or repair rules for invalid actions.",
            "- Transition: how an action changes the current partial solution, inventory, capacity, time, position, budget, or other state variables.",
            "- Reward: aligned with the official objective/evaluation metric; include penalties for constraint violations and optional dense shaping that preserves the final objective.",
            "- Terminal condition: when an episode ends, including success, infeasible dead-end, timeout, horizon, or completed assignment.",
            "- Evaluation protocol: validation scenarios, seeds, baseline comparison, constraint violation rate, and final objective/metric.",
            "",
            "**Algorithm recommendations by action/data setting:**",
            "- Discrete online RL: DQN/Rainbow DQN for manageable discrete actions; Rainbow combines Double DQN, prioritized replay, dueling networks, multi-step returns, distributional RL, and noisy exploration.",
            "- General online RL: PPO is a robust default when the environment is available and actions may be discrete or continuous; add action masks/feasibility repair for combinatorial tasks.",
            "- Continuous online RL: SAC is often a strong sample-efficient off-policy default; TD3 is a strong deterministic actor-critic baseline.",
            "- Offline RL: start with behavior cloning as a sanity baseline, then CQL, IQL, TD3+BC, or Decision Transformer when logged trajectories are available.",
            "- Model-based/world-model methods such as Dreamer-style agents are powerful but usually too complex unless the task provides a rich simulator and enough runtime.",
            "",
            "**Implementation expectations if RL is chosen:**",
            "- Implement a Gymnasium-like environment with `reset(seed=None, options=None)` and `step(action)` returning `(obs, reward, terminated, truncated, info)`.",
            "- Keep reward calculation and constraint checking as deterministic, testable functions.",
            "- For combinatorial actions, expose `valid_action_mask` or repair invalid actions before training.",
            "- Always compare RL against simple OR/heuristic baselines; if RL is slower or less stable, keep the stronger non-RL solver.",
            "",
        ]
    }


ROBUSTNESS_GENERALIZATION_STRATEGY = {
    "💡 Recommendation: Robustness & Generalization Strategy": [
        "",
        "**To improve model robustness and generalization on unseen data:**",
        "",
        "✅ **Architecture**: Match model inductive bias to data structure (e.g., CNNs/ViTs for spatial grids, Transformers/RNNs for sequences, GNNs/GCNs for graphs/topology)",
        "✅ **Input Strategy**: Handle variable-length or large-scale inputs via **windowing strategies** or patch-based processing (consider overlap for smoother predictions)",
        "✅ **Regularization**: Consider using Dropout, Batch/Layer Norm, Weight Decay, or Label Smoothing",
        "✅ **Loss Function**: Inspect class distribution and adapt loss accordingly (e.g., weighted loss, FocalLoss, or task-specific objectives)",
        "✅ **Learning Rate**: Consider using adaptive schedules like Cosine Annealing or ReduceLROnPlateau or Warmup with differential rates if needed",
        "✅ **Data Augmentation**: Apply domain-appropriate augmentation based on data modality (e.g., geometric transforms, masking, mixup)",
        "✅ **Validation**: Monitor validation metrics strictly and use early stopping to prevent overfitting",
        "",
        "⚠️ **Note**:",
        "Prioritize capturing the intrinsic structure of the data (Inductive Bias) over simply increasing model size.",
        "",
    ]
}


def prompt_leakage_prevention():
    """Data leakage prevention."""
    return {
        "🚨 DATA LEAKAGE PREVENTION": [
            "",
            "⚠️ **Strict Isolation Principle**: Validation/Test data must remain strictly unseen during training.",
            "",
            "✅ **Sequence**: Always **Split Data FIRST**, then apply processing.",
            "✅ **Stateful Transformations**: Fit all Scalers, Encoders, Imputers, and Tokenizers **ONLY on Training data**, then use `.transform()` on Validation/Test.",
            "✅ **Feature Engineering**: Calculate global statistics (e.g., mean, variance, vocabulary) solely from the Training set.",
            "✅ **Target Leakage**: Never use target information (e.g., Target Encoding) from the validation set.",
            "",
        ]
    }


def prompt_resp_fmt():
    """Response format for plan + code"""
    return {
        "Response format": (
            "Your response should be a brief outline/sketch of your proposed solution in natural language, "
            "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
            "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
        )
    }


def get_internet_clarification(pretrain_model_dir: str = ""):
    """Internet access clarification for improve/debug stages."""
    lines = [
        "**⚠️ IMPORTANT: Internet Access During Code Development**",
        "- The \"no internet access\" restriction mentioned in the task description applies **ONLY to submission evaluation after code generation** (for mle-bench test set).",
        "- **During code development, you CAN and SHOULD use online resources** such as torch.hub.load(), HuggingFace transformers, timm, etc.",
    ]
    if pretrain_model_dir:
        lines.append(
            f"- **Model paths under `{pretrain_model_dir}/` are GUARANTEED to exist and be available** (e.g., DINOv3, Siglip2 etc.). You can directly use them without `Path question`."
        )
    lines.append(
        "- **Do NOT question internet access concerns - all standard ML libraries and pretrained models are available during development."
    )
    return lines
