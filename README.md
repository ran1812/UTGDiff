# Instruction-Based Molecular Graph Generation with Unified Text-Graph Diffusion Model (UTGDiff)

This is the code for the Paper: Instruction-Based Molecular Graph Generation with Unified Text-Graph Diffusion Model

![Overview of UTGDiff](./overview_final.png)

## Environment setup

The basic environment requirement is pytorch, here's an example for environment setup:

```
cd ./text-graph-diffusion/
conda create -n UTGDiff python=3.10
conda activate UTGDiff
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt 
```

All the model checkpoint is saved at https://drive.google.com/drive/folders/18EqQ7MDHesmtiMiZz2o09PyeSwyf0hXb?usp=drive_link

## Generation

The generation code is under the ./generation dir

```
cd ./generation/
```

To train the model from roberta-base, run the below command directly:

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29501 main_dist.py
```

To train the model from the pretrain model roberta-base, run the below command directly:

To load the pretrain checkpoint, you can put the files in ./pretrain_model under the ./generation folder, then run:

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29501 main_dist_pretrain.py
```

The generation & editing results is saved at ./generation/generation_results and ./editing/generation_results


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

