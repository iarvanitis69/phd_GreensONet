# paddle
export CUDA_VISIBLE_DEVICES=4
python main_diffusion_plate.py

export CUDA_VISIBLE_DEVICES=3
python main_poisson.py

# Max Memory 44.95 GB
export CUDA_VISIBLE_DEVICES=0
python main_stokes.py

export CUDA_VISIBLE_DEVICES=1
python main_diffusion_pipe.py

export CUDA_VISIBLE_DEVICES=3
python Inference.py

# pipe
export CUDA_VISIBLE_DEVICES=4
python -u main_diffusion_pipe.py \
    2>&1 | tee train_0904_pipe.log

# torch
export CUDA_VISIBLE_DEVICES=1
python -u main.py \
    --epochs 5000 \
    2>&1 | tee train_0903.log

export CUDA_VISIBLE_DEVICES=2
python Inference.py