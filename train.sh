export EXP_NAME="molmoact2-ft-ah-only"

cd "$(dirname "$0")/experiments"

echo "TensorBoard logs: /tmp/molmoact2_test/${EXP_NAME}/tensorboard"

python -m torch.distributed.run \
  --nnodes="${NNODES:-1}" --nproc-per-node=8 \
  --node_rank="${RANK:-0}" --master_addr="${ADDR:-127.0.0.1}" --master_port="${PORT:-29415}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  piper_x \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 \
  --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="/tmp/molmoact2_test/${EXP_NAME}" \
  --packing=false \
  --dynamic_seq_len=true \
  --img_resize=320x180 \
  --crop_mode=resize \
  --ft_vlm=true \
  --ft_action_expert=true \
  --ft_embedding=lm_head \
  --lora_enable=true \
  --lora_rank=8 \
  --add_action_tokens=false \
  --add_depth_tokens=false \
  --llm_learning_rate=5e-5 \
  --vit_learning_rate=5e-5 \
  --connector_learning_rate=5e-5 \
  --action_expert_learning_rate=5e-5
