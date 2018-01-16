source ~/tensorflow/bin/activate      # bash, sh, ksh, or zsh
CUDA_VISIBLE_DEVICES=1 python stock_predict_6.py --num_layers=2 --hidden_size=30 training_steps=50000 # Uses GPU 1.

