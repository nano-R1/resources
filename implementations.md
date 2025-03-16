# RL Implementations

## LLM RL Repos
**TRL-based:**
- [TRL](https://github.com/huggingface/trl/tree/main/trl)
    - Algorithms: PPO, GRPO, KTO, RLOO, (O)DPO, XPO, others
    - Backends: DDP, accelerate (FSDP, DeepSpeed ZeRO 1/2/3)
    - Inference: transformers, vLLM
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)
    - Algorithms: same as TRL
    - Backends: accelerate, FSDP + QLoRA, Ray
- [Unsloth](https://github.com/unslothai/unsloth)
    - Algorithms: same as TRL
    - Backends: custom (single-GPU PEFT, memory-optimized)
- [rStar](https://github.com/microsoft/rStar)
    - Algorithms: rStar
    - Inference: vLLM
- [LlamaGym](https://github.com/KhoomeiK/LlamaGym)
    - Algorithms: PPO (multi-turn)
    - Inference: transformers
- [verifiers](https://github.com/willccbb/verifiers)
    - Algorithms: GRPO (multi-turn)
    - Inference: vLLM
- [groundlight/r1-vlm](https://github.com/groundlight/r1_vlm)
    - Algorithms: GRPO (multimodal)
    - Inference: vLLM

**veRL-based:**
- [veRL](https://github.com/volcengine/verl)
    - Algorithms: PPO, GRPO, PRIME
    - Backends: FSDP, Megatron-LM
    - Inference: transformers, vLLM, SGLang ([open PR](https://github.com/volcengine/verl/pull/490))
- [RAGEN](https://github.com/ZihanWang314/RAGEN)
    - Algorithms: PPO + RICO (multi-turn)
- [PRIME](https://github.com/PRIME-RL/PRIME)
    - Algorithms: PRIME

**OpenRLHF-based:**
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
    - Algorithms: PPO, GRPO, DPO, KTO, RLOO, REINFORCE++
    - Backends: Ray 
    - Inference: vLLM
- [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)
    - Algorithms: PPO
- [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher)
    - Algorithms: REINFORCE++

**torchtune-based:**
- [torchtune](https://github.com/pytorch/torchtune)
    - Algorithms: DPO, PPO, GRPO
- [OpenPipe/deductive-reasoning](https://github.com/OpenPipe/deductive-reasoning)
    - Algorithms: GRPO (KL-free)


**torch (standalone)**
- [oat](https://github.com/sail-sg/oat/tree/main)
    - Algorithms: PPO, (O)DPO, XPO
    - Backends: accelerate (DeepSpeed)
    - Inference: vLLM + Mosec
- [allenai/open-instruct](https://github.com/allenai/open-instruct)
    - Algorithms: PPO, DPO, GRPO
    - Backends: accelerate
    - Inference: vLLM
- [VinePPO/treetune](https://github.com/McGill-NLP/VinePPO)
    - Algorithms: PPO, DPO, VinePPO, RestEM
- [Lamorel](https://github.com/flowersteam/lamorel/tree/main)
    - Algorithms: PPO
    - Backends: accelerate
    - Inference: transformers
- [ReMax](https://github.com/liziniu/ReMax)
    - Algorithms: ReMax

**MLX**
- [mlx-lm](https://github.com/ml-explore/mlx-examples/blob/main/llms/README.md)
    - Algorithms: DPO ([open PR](https://github.com/ml-explore/mlx-examples/pull/1279))

**Jax/Flax**
- Nothing?

## Deep RL Repos (non-LLM)

**Jax/Flax**
- [JAX-PPO](https://github.com/zombie-einstein/JAX-PPO)

**MLX**
- [clean-rl-mlx](https://github.com/andrew-silva/clean-rl-mlx)


