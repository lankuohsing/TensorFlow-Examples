source ~/tensorflow/bin/activate      # bash, sh, ksh, or zsh
CUDA_VISIBLE_DEVICES=0 python stock_predict_5.py --num_layers=2 --hidden_size=20 training_steps=10000 # Uses GPU 0.

