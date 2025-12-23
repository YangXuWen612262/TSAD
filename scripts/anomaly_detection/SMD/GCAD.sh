export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id GCAD_SMD \
  --model GCAD \
  --data SMD \
  --features M \
  --seq_len 100 \
  --pred_len 1 \
  --d_model 128 \
  --e_layers 3 \
  --enc_in 38 \
  --c_out 38 \
  --batch_size 128 \
  --train_epochs 10 \
  --learning_rate 0.001 \
  --patience 3