# Vosh

This repository contains a PyTorch implementation of the paper: [Voxel-Mesh Hybrid Representation for Real-Time View Synthesis](https://arxiv.org/abs/2403.06505).

### [Project Page](https://zyyzyy06.github.io/Vosh/) | [Arxiv](https://arxiv.org/abs/2403.06505) | [Paper]()

![](assets/teaser.png)

# Install

```bash
git clone https://github.com/zachzhang07/vosh.git
cd vosh
```

### Install with pip
```bash
conda create -n vosh python==3.8.13
conda activate vosh

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install -r requirements.txt

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

```

<!-- ### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
However, this may be inconvenient sometimes.
Therefore, we also provide the `setup.py` to build each extension:
```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
``` -->

# Usage

We majorly support COLMAP dataset like [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).
Please download and put them under `../data/`.

For custom datasets:
```bash
# prepare your video or images under /data/custom, and run colmap (assumed installed):
python scripts/colmap2nerf.py --video ../data/custom/video.mp4 --run_colmap # if use video
python scripts/colmap2nerf.py --images ../data/custom/images/ --run_colmap # if use images
```

### Basics
First time running will take some time to compile the CUDA extensions.
```bash
## train and eval
# mip-nerf 360
python main_vol.py ../data/360_v2/bicycle/ --workspace ../output/bicycle --contract
python main_mesh.py ../data/360_v2/bicycle/ --vol_path ../output/bicycle \
  --workspace ../output/bicycle_mesh
python main_vosh.py ../data/360_v2/bicycle/ --vol_path ../output/bicycle_mesh --workspace ../output/bicycle_base --lambda_mesh_weight 0.001 --mesh_select 0.9 --keep_center 0.25 --lambda_bg_weight 0.01
python main_vosh.py ../data/360_v2/bicycle/ --vol_path ../output/bicycle_mesh --workspace ../output/bicycle_light --lambda_mesh_weight 0.01 --mesh_select 1.0 --keep_center 0.25 --lambda_bg_weight 0.01 --use_mesh_occ_grid --mesh_check_ratio 8
```
If you want to eval Vosh in 7 scenes of mip-nerf 360 dataset, just run:
```bash
python full_eval_360.py ../data/360_v2/ --workspace ../output/
```

Please check full_eval_360.py for different hyper-parameters of different kind of scenes, and check `main_*.py` for all options.

### Acknowledgement
Heavily borrowed from [torch-merf](https://github.com/ashawkey/torch-merf) and [nerf2mesh](https://github.com/ashawkey/nerf2mesh). Many thanks to Jiaxiang.


# Citation

```
@article{zhang2024vosh,
  title={Vosh: Voxel-Mesh Hybrid Representation for Real-Time View Synthesis},
  author={Zhang, Chenhao and Zhou, Yongyang and Zhang, Lei},
  journal={arXiv preprint arXiv:2403.06505},
  year={2024}
}
```