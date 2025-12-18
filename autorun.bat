@echo off
:: ====================================================================
:: Windows Batch Script for Autoformer Long-Term Forecasting Training
:: ====================================================================

:: 1. 激活你的 Anaconda 环境
::    请将下面的 "your_env_name" 替换为你实际的环境名称 (例如: pytorch_cu116)
call conda activate tslib

:: 检查 conda 环境是否激活成功
if %errorlevel% neq 0 (
    echo.
    echo *******************************************************************
    echo *  错误: Anaconda 环境未能激活!                            *
    echo *  请确认 'your_env_name' 已被替换为你正确的环境名.        *
    echo *******************************************************************
    echo.
    goto :eof
)

:: 2. 设置环境变量
::    设置使用的GPU索引 (0代表第一张卡, 1代表第二张, 以此类推)
set CUDA_VISIBLE_DEVICES=0

::    设置模型名称变量
set model_name=Autoformer

echo Using model: %model_name%
echo Using GPU: %CUDA_VISIBLE_DEVICES%
echo.

:: ====================================================================
:: 任务 1: 预测长度为 96
:: ====================================================================
echo Running training for pred_len = 96 ...
python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/ETT-small/ ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_96_96 ^
  --model %model_name% ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 96 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 7 ^
  --dec_in 7 ^
  --c_out 7 ^
  --des "Exp" ^
  --itr 1

:: ====================================================================
:: 任务 2: 预测长度为 192
:: ====================================================================
echo Running training for pred_len = 192 ...
python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/ETT-small/ ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_96_192 ^
  --model %model_name% ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 192 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 7 ^
  --dec_in 7 ^
  --c_out 7 ^
  --des "Exp" ^
  --itr 1

:: ====================================================================
:: 任务 3: 预测长度为 336
:: ====================================================================
echo Running training for pred_len = 336 ...
python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/ETT-small/ ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_96_336 ^
  --model %model_name% ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 336 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 7 ^
  --dec_in 7 ^
  --c_out 7 ^
  --des "Exp" ^
  --itr 1

:: ====================================================================
:: 任务 4: 预测长度为 720
:: ====================================================================
echo Running training for pred_len = 720 ...
python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/ETT-small/ ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_96_720 ^
  --model %model_name% ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 720 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 7 ^
  --dec_in 7 ^
  --c_out 7 ^
  --des "Exp" ^
  --itr 1

echo.
echo All tasks finished.
pause

