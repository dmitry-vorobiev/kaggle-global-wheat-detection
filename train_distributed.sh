export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=2 --use_env src/train.py
