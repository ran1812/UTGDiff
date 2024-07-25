## Generation

Train:

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29501 main_dist.py

or 

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29501 main_dist_pretrain.py

sampling:

python predict_downstream_dist.py
python aro.py

eval:

python eval.py

## Editing

Train:

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29501 main_dist_pretrain.py

sampling:

python predict_downstream_dist.py

eval:

python eval_MOIretro.py
python eval_MOIfp.py

