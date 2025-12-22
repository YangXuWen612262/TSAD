export CUDA_VISIBLE_DEVICES=3

python -u run.py \
      --task_name anomaly_detection \
      --is_training 1 \
      --model_id PSM_OracleAD \
      --model OracleAD \
      --root_path ./dataset/PSM \
      --data_path PSM.csv \
      --data PSM \
      --features M \
      --seq_len 10 \
      --pred_len 0 \
      --d_model 64 \
      --e_layers 2 \
      --d_layers 2 \
      --enc_in 25 \
      --batch_size 1024 \
      --train_epochs 20 \
      --learning_rate 0.0005 \
      --lambda_recon 0.1 \
      --lambda_dev 3.0