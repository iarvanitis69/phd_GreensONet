# Max Memory 19.69 GB
export CUDA_VISIBLE_DEVICES=4
python main_diffusion_plate.py

# Max Memory 1 GB
export CUDA_VISIBLE_DEVICES=3
python main_poisson.py

# Max Memory 44.95 GB
export CUDA_VISIBLE_DEVICES=0
python main_stokes.py

# Max Memory 8.63 GB
export CUDA_VISIBLE_DEVICES=1
python main_diffusion_pipe.py

export CUDA_VISIBLE_DEVICES=3
python Inference.py
