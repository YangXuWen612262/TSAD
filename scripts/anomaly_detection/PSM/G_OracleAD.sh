export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM_GOracleAD \
  --model G_OracleAD \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 25 \
  --c_out 25 \
  --batch_size 128 \
  --train_epochs 3 \
  --anomaly_ratio 1 \
  --learning_rate 1e-4 \
  --dropout 0.1 \
  --d_model 128 \
  --lambda_pred 0.1 \
  --lambda_grad 0.05 \
  --beta_grad 4.0 \
  --use_grad_soft 1 \
  --dist_norm 1 \
  --score_point last \
  --patience 3 \
  --log_step 100
