# Open Questions

Questions to resolve in pursuit of an optimal recipe for reasoning-oriented RL with LLMs

## Physics of LLM RL

- At what **base**  model scale (in parameters) does RL for reasoning become minimally viable (e.g. >50% on GSM8K)?
- At what **instruct**  model scale (in parameters) does RL for reasoning become minimally viable (e.g. >50% on GSM8K)?
- At what model scale does **outcome-only** RL for reasoning become more computationally 
- Does the presence of more LLM-generated chain-of-thought traces on the web from 2022-2024 explain (in part) the emergence of increased CoT during verifiable RL (as in R1)?
- Under what conditions does RL become "better" than distillation from reasoning models like R1?
- Do RL-trained reasoning models generalize OOD better than 
- What factors influence the OOD generalization capabilities of RL-trained reasoning models?

## GRPO Implementation Details

- How do off-policy steps affect training stability? 
    - Note: if training is stable one step off-policy for all rollouts, then rollout inference + model updates can be fully overlapped in theory
- What are the scaling laws for group size + batch size? 

## PPO Implementation Details

## PPO vs. GRPO vs. RLOO vs. PRIME vs. KTO vs. DPO vs. ... 

## Multi-Turn + Agentic RL

## Multi-Agent RL


