![1_low_res](https://github.com/user-attachments/assets/3a1efa45-b489-494a-93f9-af1517b49563)# SMPL-Pose: Adaptive graph transformer with llm priors for 3D human reconstruction

## 1. Overview
SMPL-Pose is a cutting-edge method for monocular 3D human shape and pose estimation. Leveraging the power of transformers, it provides accurate and efficient solutions for 3D human modeling tasks. This repository contains the code, datasets, and instructions for using SMPL-Pose.

## 2. Hardware Requirements
- **Testing**: Most modern GPUs are sufficient to run the testing process.
- **Training**: For optimal training performance, it is highly recommended to use 2 NVIDIA A100 GPUs.

## 3. Installation
### 3.1 Create Conda Environment
```bash
conda create -n SMPL-Pose python=3.8
conda activate SMPL-Pose
```
### 3.2 Install Packages
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install scipy==1.5.0 scikit-image==0.19.1 opencv-python==4.5.4.58 imageio matplotlib numpy==1.20.3 chumpy==0.70 ipython ipykernel ipdb smplx==0.1.28 tensorboardx==2.4 tensorboard==2.7.0 easydict pillow==8.4.0
```
### 3.3 Install Pytorch3D
```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
wget https://anaconda.org/pytorch3d/pytorch3d/0.5.0/download/linux-64/pytorch3d-0.5.0-py38_cu111_pyt180.tar.bz2 --no-check-certificate
conda install pytorch3d-0.5.0-py38_cu111_pyt180.tar.bz2
rm pytorch3d-0.5.0-py38_cu111_pyt180.tar.bz2
```

## 4. How to Run
### 4.1 Prepare Data
- Download the meta data and extract it into `PATH_to_SMPL-Pose/meta_data`.
- Download the pretrained models and extract it into `PATH_to_SMPL-Pose/pretrained`.

### 4.2 Run Demo
```bash
python demo.py --img_path samples/im01.png
```

## 5. Train and Test
### 5.1 Prepare Datasets
There are two ways to download the datasets:
- **Recommended (faster)**: Use `azcopy`.
    1. Download `azcopy` from [here](Please replace with the actual download link).
    2. Navigate to the directory where you want to store the dataset: `cd PATH_to_STORE_DATASET`
    3. Set the `azcopy` path: `azcopy_path=PATH_to_AZCOPY`
    4. Run the download script: `bash PATH_to_SMPL-Pose/scripts/download_datasets_azcopy.sh`
    5. Create a symbolic link: `cd PATH_to_SMPL-Pose && ln -s PATH_to_STORE_DATASET ./datasets`
- **Alternative**: Use `wget` (usually slower and less stable, but no dependency on `azcopy`).
    1. Navigate to the dataset storage directory: `cd PATH_to_STORE_DATASET`
    2. Run the download script: `bash PATH_to_SMPL-Pose/scripts/download_datasets_wget.sh`

### 5.2 Test
- **Test on H36M dataset**
    - For SMPL-Pose:
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --eval_only --val_batch_size=128 --model_type=SMPL-Pose --data_mode=h36m --hrnet_type=w32 --load_checkpoint=pretrained/SMPL-Pose_h36m.pt 
```
    - For SMPL-Pose:
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --eval_only --val_batch_size=128 --model_type=SMPL-Pose --data_mode=h36m --hrnet_type=w48 --load_checkpoint=pretrained/SMPL-Pose-L_h36m.pt 
```
- **Test on 3DPW dataset**
    - For SMPL-Pose:
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --eval_only --val_batch_size=128 --model_type=SMPL-Pose --data_mode=3dpw --hrnet_type=w32 --load_checkpoint=pretrained/SMPL-Pose_3dpw.pt 
```
    - For SMPL-Pose-L:
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --eval_only --val_batch_size=128 --model_type=SMPL-Pose --data_mode=3dpw --hrnet_type=w48 --load_checkpoint=pretrained/SMPL-Pose-L_3dpw.pt 
```

### 5.3 Train
- **For SMPL-Pose**:
    1. **Train CNN backbone on mixed data**:
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --exp_name=backbone --batch_size=100 --num_workers=8 --lr=2e-4 --data_mode=h36m --model_type=backbone --num_epochs=50 --hrnet_type=w32  
```
    2. **Train SMPL-Pose on mixed data**:
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --exp_name=SMPL-Pose --batch_size=100 --num_workers=8 --lr=2e-4 --data_mode=h36m --model_type=SMPL-Pose --num_epochs=100 --hrnet_type=w32 --load_checkpoint=logs/backbone/checkpoints/epoch_049.pt
```
    3. **Finetune SMPL-Pose on 3DPW**:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --exp_name=SMPL-Pose_3dpw --batch_size=32 --num_workers=8 --lr=1e-4 --data_mode=3dpw --model_type=SMPL-Pose --num_epochs=2 --hrnet_type=w32 --load_checkpoint=logs/SMPL-Pose/checkpoints/epoch_***.pt --summary_steps=100
```
- **For SMPL-Pose-L**:
    1. **Train CNN backbone on mixed data**:
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --exp_name=backbone-L --batch_size=100 --num_workers=8 --lr=2e-4 --data_mode=h36m --model_type=backbone --num_epochs=50 --hrnet_type=w48  
```
    2. **Train SMPL-Pose-L on mixed data**:
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --exp_name=SMPL-Pose-L --batch_size=100 --num_workers=8 --lr=2e-4 --data_mode=h36m --model_type=SMPL-Pose --num_epochs=100 --hrnet_type=w48 --load_checkpoint=logs/backbone-L/checkpoints/epoch_049.pt
```
    3. **Finetune SMPL-Pose-L on 3DPW**:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --exp_name=SMPL-Pose-L_3dpw --batch_size=32 --num_workers=8 --lr=1e-4 --data_mode=3dpw --model_type=SMPL-Pose --num_epochs=2 --hrnet_type=w48 --load_checkpoint=logs/SMPL-Pose-L/checkpoints/epoch_***.pt --summary_steps=100
```

## 6. Cite
If you use SMPL-Pose in your work, please cite the following paper:
```bibtex
@article{xu2024SMPL-Pose,
  title={SMPL-Pose: Taming Transformers for Monocular 3D Human Shape and Pose Estimation},
  author={Xu, Xiangyu and Liu, Lijuan and Yan, Shuicheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```

## 7. Related Resources
Explore these related resources to deepen your understanding of 3D human modeling:
- METRO
- Mesh Graphormer
- RSC-Net
- Texformer
- Sewformer
- GP-NeRF
