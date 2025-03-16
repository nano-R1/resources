# RL Implementations

## LLM RL Repos
**TRL-based:**
- [TRL](https://github.com/huggingface/trl/tree/main/trl)
    - Algorithms: PPO, GRPO, KTO, RLOO, (O)DPO, XPO, others
    - Backends: DDP, accelerate (FSDP, DeepSpeed ZeRO 1/2/3)
    - Inference: Transformers, vLLM
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)
    - Algorithms: same as TRL
    - Backends: accelerate, FSDP + QLoRA, Ray
- [Unsloth](https://github.com/unslothai/unsloth)
    - Algorithms: same as TRL
    - Backend: custom (single-GPU PEFT, memory-optimized)
- [verifiers](https://github.com/willccbb/verifiers)
    - Algorithms: GRPO (multi-turn)
    - Backends: DDP, accelerate (FSDP, DeepSpeed ZeRO 1/2/3)
    - Inference: vLLM
- [groundlight/r1-vlm](https://github.com/groundlight/r1_vlm)
    - Algorithms: GRPO: (multimodal)

**veRL-based:**
- [veRL](https://github.com/volcengine/verl)
    - Algorithms: PPO, GRPO, PRIME
    - Backends: FSDP, Megatron-LM
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- [PRIME](https://github.com/PRIME-RL/PRIME)

**torchtune-based:**
- [torchtune](https://github.com/pytorch/torchtune)
    - Algorithms: DPO, PPO, GRPO
- [OpenPipe/deductive-reasoning](https://github.com/OpenPipe/deductive-reasoning)
    - Algorithms: GRPO (KL-free)


**torch (standalone)**
- [oat](https://github.com/sail-sg/oat/tree/main)

- [open-instruct](https://github.com/allenai/open-instruct)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
    - Algorithms: PPO, GRPO, DPO, KTO, RLOO, REINFORCE++
- [VinePPO/treetune](https://github.com/McGill-NLP/VinePPO)
    - Algorithms: PPO, DPO, VinePPO, RestEM

**MLX**
- [mlx-lm](https://github.com/ml-explore/mlx-examples/blob/main/llms/README.md)
    - Algorithms: DPO ([open PR](https://github.com/ml-explore/mlx-examples/pull/1279))


**Jax/Flax**
- Nothing?

## Deep RL Repos (non-LLM)

**Jax/Flax**
- [JAX-PPO](https://github.com/zombie-einstein/JAX-PPO)
    - Note: tabula rasa / non-LLM

**MLX**
- [clean-rl-mlx](https://github.com/andrew-silva/clean-rl-mlx)


