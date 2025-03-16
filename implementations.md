# RL Implementations

## LLM RL Repos
**TRL-based:**
- [TRL](https://github.com/huggingface/trl/tree/main/trl)
    - Algorithms: PPO, GRPO, KTO, RLOO, (O)DPO, XPO, others
    - Backends: DDP, accelerate (FSDP, DeepSpeed ZeRO 1/2/3)
    - Inference: Transformers, vLLM
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)
- [Unsloth](https://github.com/unslothai/unsloth)
    - Algorithms: same as TRL
    - Backend: custom (single-GPU PEFT, memory-optimized)
- [verifiers](https://github.com/willccbb/verifiers)
    - Algorithms: GRPO (multi-turn)
    - Backends: DDP, accelerate (FSDP, DeepSpeed ZeRO 1/2/3)
    - Inference: vLLM

**veRL-based:**
- [veRL](https://github.com/volcengine/verl)
    - Algorithms: PPO, GRPO, PRIME
    - Backends: FSDP, Megatron-LM
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- [PRIME](https://github.com/PRIME-RL/PRIME)

**torchtune-based:**
- [torchtune](https://github.com/pytorch/torchtune)
- [OpenPipe deductive-reasoning](https://github.com/OpenPipe/deductive-reasoning)

**torch (standalone)**
- [oat](https://github.com/sail-sg/oat/tree/main)
- [open-instruct](https://github.com/allenai/open-instruct)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

**MLX**
- [mlx-lm](https://github.com/ml-explore/mlx-examples/blob/main/llms/README.md)
    - Algorithms: DPO ([open PR](https://github.com/ml-explore/mlx-examples/pull/1279))


## Deep RL Repos (non-LLM)

**jax/flax**
- [JAX-PPO](https://github.com/zombie-einstein/JAX-PPO)
    - Note: tabula rasa / non-LLM
- [clean-rl-mlx](https://github.com/andrew-silva/clean-rl-mlx)


