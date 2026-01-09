export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM_OracleAD \
  --model OracleAD \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 25 \
  --c_out 25 \
  --d_model 128 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --lambda_recon 1 \
  --lambda_dev 0.1 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3 \
  --patience 3
