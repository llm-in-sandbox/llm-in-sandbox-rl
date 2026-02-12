set -x

# clean up before running (optional, recommended if previous run was interrupted)
docker ps -aq --filter 'ancestor=cdx123/llm-in-sandbox:v0.1' | xargs -r docker rm -f
pkill -9 -f "llm-in-sandbox benchmark" 2>/dev/null

# Clean up
ray stop --force 2>/dev/null || true
sleep 2
pkill -9 ray 2>/dev/null || true
pkill -9 wandb 2>/dev/null || true
sleep 2

# Environment variables
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_LOGGING_LEVEL=CRITICAL  # Suppress vllm tool parser errors
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

TRAIN_FILES="['./data/llm_sandbox_instruct_pretrain/train_verl.json']"
# Validate on a subset of tasks (add more by uncommenting the full list below)
VAL_FILES="['./data/llm_sandbox_math_mini/test_verl.json','./data/llm_sandbox_biomed_mini/test_verl.json','./data/llm_sandbox_long_context_mini/test_verl.json']"
# VAL_FILES="['./data/llm_sandbox_math_mini/test_verl.json','./data/llm_sandbox_biomed_mini/test_verl.json','./data/llm_sandbox_long_context_mini/test_verl.json','./data/llm_sandbox_chem_mini/test_verl.json','./data/llm_sandbox_physics_mini/test_verl.json','./data/llm_sandbox_instruct_follow_mini/test_verl.json']"
# NOTE: Online validation uses Rouge (long-context) and math-verify (physics) for speed.
# Evaluate the final model on the full benchmark: https://github.com/llm-in-sandbox/llm-in-sandbox/blob/main/llm_in_sandbox/benchmark/README.md

# Model and training config
DOCKER_IMAGE="cdx123/llm-in-sandbox:v0.1"
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
PROMPT_LENGTH=16384
RESPONSE_LENGTH=65536
MAX_STEPS=100
USP=1
TSP=1
OVERLONG_FILTER=false
OVERLONG_PENALTY=true
EXP_NAME=llm_sandbox_$(basename $MODEL_PATH)
CKPTS_DIR=./exp/${EXP_NAME}

python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=8 \
    data.val_batch_size=512 \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=128 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( (PROMPT_LENGTH + RESPONSE_LENGTH + 1) / USP )) \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${USP} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TSP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='LLM-in-Sandbox' \
    trainer.experiment_name=${EXP_NAME} \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.resume_mode=auto \
    rllm.env.name=llm_in_sandbox \
    rllm.agent.name=llm_in_sandbox_agent \
    rllm.agent.max_steps=${MAX_STEPS} \
    rllm.agent.overlong_filter=${OVERLONG_FILTER} \
    rllm.agent.trajectory_timeout=5400 \
    rllm.accumulate_reasoning=false \
    rllm.agent.overlong_step_penalty=${OVERLONG_PENALTY} \
    +rllm.env.env_args.docker_image=${DOCKER_IMAGE} \
    +rllm.tool_parser_type=hermes \
    trainer.total_epochs=1000
