export WANDB_PROJECT="llm-verify-mech"
export WANDB_ENTITY="yefan_zhou"

# # Run 1 on GPUs 0-3 (background)
# LOG_DIR=/data/yefan/llm-verify-mech/trl/verifier_training/logs/sft_config_run1
# mkdir -p $LOG_DIR
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate_config.yaml --main_process_port 29500 \
#     sft_lora.py --config sft_config_run1.yaml 2>&1 | tee $LOG_DIR/train.log &

# # Run 2 on GPUs 4-7 (background)
# LOG_DIR=/data/yefan/llm-verify-mech/trl/verifier_training/logs/sft_config_run1_prompt1
# mkdir -p $LOG_DIR
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file accelerate_config.yaml --main_process_port 29501 \
#     sft_lora.py --config sft_config_run1_prompt1.yaml 2>&1 | tee $LOG_DIR/train.log &

# # Wait for both to finish
# wait

# # Run 3 on GPUs 0-3
# LOG_DIR=/data/yefan/llm-verify-mech/trl/verifier_training/logs/sft_config_run2
# mkdir -p $LOG_DIR
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate_config.yaml --main_process_port 29500 \
#     sft_lora.py --config sft_config_run2.yaml 2>&1 | tee $LOG_DIR/train.log &

# Run 4 on GPUs 4-7
LOG_DIR=/data/yefan/llm-verify-mech/trl/verifier_training/logs/sft_config_run2_prompt1
mkdir -p $LOG_DIR
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file accelerate_config.yaml --main_process_port 29501 \
    sft_lora.py --config sft_config_run2_prompt1.yaml 2>&1 | tee $LOG_DIR/train.log &

wait
echo "All 4 runs complete"