export EXP_NAME="molmoact2-test"

cd "$(dirname "$0")/experiments"

echo "TensorBoard logs: /tmp/molmoact2_test/${EXP_NAME}/tensorboard"
echo "Set DEBUG=true for a tiny smoke-test model instead of loading full MolmoAct2."

python -m torch.distributed.run \
  --nnodes="${NNODES:-1}" --nproc-per-node=1 \
  --node_rank="${RANK:-0}" --master_addr="${ADDR:-127.0.0.1}" --master_port="${PORT:-29415}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  piper_x \
  --debug="${DEBUG:-false}" \
  --max_duration=5 \
  --device_batch_size=1 \
  --global_batch_size=1 \
  --num_workers=0 \
  --data.timeout=0 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=0 \
  --save_folder="/tmp/molmoact2_test/${EXP_NAME}" \
  --packing=false \
  --dynamic_seq_len=true \
  --img_resize=224x224 \
  --crop_mode=resize \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none \
  --lora_enable=false \
  --lora_rank=8 \
  --add_action_tokens=false \
  --add_depth_tokens=false \
  --llm_learning_rate=5e-5 \
  --vit_learning_rate=5e-5 \
  --connector_learning_rate=5e-5 \
  --action_expert_learning_rate=5e-5 \
  precision=amp_bf16 \
  fsdp.precision=pure \
  compile_loss=false \
  save_final_optim=false \
  save_merged_lora_checkpoint=false
