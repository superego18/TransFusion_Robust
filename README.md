# Setting Environment

## Requirement of OS and Graphics card

### OS: Ubuntu 18.04 (I used Windows 10 WSL version)
- CUDA 10.1 supports 14.04 ~ 18.10.
- And it looks like CUDA 11.1 only supports 16.04 ~ 20.04.
- In my experience, 22.04 was not possible. 16.04 ~ 20.04 is the expected range of possibilities, but I highly recommend going with 18.04.

### Graphics Card (=VGA)
- **Nivida GeForce GTX 1060 Ti (compute capability = 7.5)**
- **Nvidia GeForce RTX 3080 Mobile (compute capability = 8.6)**
- Please refer [site_link](https://forums.developer.nvidia.com/t/cuda-compatibility-between-nvidia-rtx-a5000-and-geforce-rtx-4060-ti/264216).
- Presumably, to be compatible with CUDA 10.1 ~ 11.1, you need to use a graphics card with a compute capability of 3.0 ~ 8.6.
- The two graphics cards above are the ones I've had success emplementing TransFusion.
- Below I'll show you how to set up your environment for TransFusion based on two types of your graphics card.

## Details

First you need to determine what type of graphics card you have. See the link above.
- **Type 1. Nvidia GeForce GTX 1060 Ti (& Those with a compute capability of 3.0 to 7.5, so CUDA 10.1 is compatible, probably.)**
- **Type 2. Nvidia GeForce RTX 3080 Mobile (& Those with a compute capability of 3.5 to 8.6, so CUDA 11.1 is compatible, probably.)**

(1) Install Python==3.7.5 on Ubuntu.

```shell
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7.5
```

(2) Install pip and virtualenv.

```shell
sudo apt install python3-pip
python3.7 -m pip install virtualenv
```

(3) Create a local directory and a virtual environment.

```shell
mkdir [your directory name] # example: project # example path: /home/chanju/project
cd [your directory name] 
virtualenv [your virtual environment name] --python=python3.7 # example: venv

# In the code below, I'll use the names I've given as examples, so check them carefully.
```

(4) Make a alias of activating virtual environment.

```shell
cd .. # go to the parent directory # example: chanju
nano .bashrc # If you don't have the nano editor, use another editor.
```
``` bash
# bash
alias activate='activate /home/chanju/project/venv/bin/activate; cd /home/chanju/project' # Add to end.
```
```shell
source .bashrc
activate
```

(5) Download cuda and cudnn.

- **If you have type1**
    - Download [cuda-10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu)
    ```shell
    # Download .run file to click download button.
    sudo sh cuda_10.1.105_418.39_linux.run
    ```
    - Download [cudnn-7.6.5](https://developer.nvidia.com/rdp/cudnn-archive)
    ```shell
    # You need login for nvidia.
    
    # Download cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.1 --> cuDNN Library for Linux
    
    tar -zxvf cudnn-10.1-linux-x64-v7.6.5.32.tgz
    
    sudo cp cuda/include/cudnn*.h /usr/local/cuda-10.1/include 
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-10.1/lib64 
    sudo chmod a+r /usr/local/cuda-10.1/include/cudnn*.h /usr/local/cuda-10.1/lib64/libcudnn*
    ```
    ```shell
    # Make path for CUDA.
    cd ~ # go to chanju repo, for example.
    nano .bashrc
    ```
    ```bash
    # bash
    # Add to end
    export PATH=/usr/local/cuda-10.1/bin:$PATH
    export LC_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LC_LIBRARY_PATH
    export LC_LIBRARY_PATH=/usr/local/cuda-10.1/extras/CUPTI/lib64:$LC_LIBRARY_PATH
    ```
    ```shell
    source .bashrc
    activate
    ```

- **If you have type2**
    - Download [cuda-11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive?)
    ```shell
    # I have tried various versions such as 11.1.0, 11.1.0 update1, etc.
    # In a successful environment, I only checked the version below, so I highly recommend below version.
    
    wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
    sudo sh cuda_11.1.0_455.23.05_linux.run
    ```
    - Download [cudnn-8.0.5](https://developer.nvidia.com/rdp/cudnn-archive)
    ```shell
    # I have tried two versions as 8.0.4, 8.0.5.
    # In a successful environment, I only checked the 8.0.5 version, so I highly recommend 8.0.5 version.
    
    # You need login for nvidia.

    # Download cuDNN v8.0.5 (November 9th, 2020), for CUDA 11.1 --> cuDNN Library for Linux (x86_64)
    
    tar -zxvf cudnn-11.1-linux-x64-v8.0.5.39.tgz
    
    sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.1/include 
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.1/lib64 
    sudo chmod a+r /usr/local/cuda-11.1/include/cudnn*.h /usr/local/cuda-11.1/lib64/libcudnn*
    ```
    ```shell
    # Make path for CUDA.
    cd ~ # go to chanju repo, for example.
    nano .bashrc
    ```
    ```bash
    # bash
    # Add to end
    export PATH=/usr/local/cuda-11.1/bin:$PATH
    export LC_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LC_LIBRARY_PATH
    export LC_LIBRARY_PATH=/usr/local/cuda-11.1/extras/CUPTI/lib64:$LC_LIBRARY_PATH
    ```
    ```shell
    source .bashrc
    activate
    ```
    
(6) Download torch and torchvision

- See [pytorch pevious version page](https://pytorch.org/get-started/previous-versions/).
- **If you have type 1**
    ```shell
    pip install torch==1.6.0+cu101 torchvision==0.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    ```
- **If you have type 2**
    ```shell
    # I have tried various versions such as 1.8.0, 1.8.1 and 1.9.0. Only 1.9.0 succeeded.
    # The other two versions showed the following error on train.py. "RuntimeError: CUDA error: no kernel image is available for execution on the device"
    # The above error seems to be a chronic problem that many people experience and could only be solved by upgrading the pytorch to 1.9.0 and reinstalling and building mmcv, mmdet, and mmdet3d.
    
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    ```

(7) Download the required packages **in advance**.

- Before we download mmcv and build mmdet and TransFusion(=mmdet3d), we need to download these.
- The versions of these packages relate to the March 2021 when mmdet3d 0.11.0 ~ 0.12.0 is released from which TransFusion was forked.
    ```shell
    pip install cython==0.29.33 setuptools==50.3.2 ninja==1.10.0post2 yapf==0.31.0 cmake==3.18.4post1 numba==0.48.0 llvmlite==0.31.0 numpy==1.19.5

    # or cmake==3.13.0, setuptools==52.0.0
    # If you need, open3d==0.17.0, protobuf==3.20.3,  

    sudo apt install python3.7-dev
    pip install pycocotools==2.0.2
    ```
    - Download [spconv-1.2.1](https://github.com/traveller59/spconv/tree/v1.2.1).
    ```shell
    git clone --branch v1.2.1 https://github.com/traveller59/spconv.git # clone v1.2.1 branch to your venv/lib/python3.7/site-packages
    python ~/project/venv/lib/python3.7/site-packages/spconv-1.2.1/setup.py bdist_wheel # Don't use python setup.py install
    ```

(8) Download mmcv-full.

- **If you have type 1**
    ``` shell
    # https://download.openmmlab.com/mmcv/dist/index.html
    # https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
    # Refer above two site, plz.

    pip install mmcv-full==1.2.4 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
    ```
- **If you have type 2**
    ``` shell
    # https://download.openmmlab.com/mmcv/dist/index.html
    # https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
    # Refer above two site, plz.

    pip install mmcv-full==1.3.10 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
    ```
    
(9) Build mmdet.

- Download tar.gz file of [v2.10.0](https://github.com/open-mmlab/mmdetection/releases/tag/v2.10.0)
``` shell
# https://mmdetection.readthedocs.io/en/v2.15.0/get_started.html
# https://github.com/open-mmlab/mmdetection/releases/tag/v2.10.0
# Refer above two site, plz.

tar -zxvf mmdetection-2.10.0.tar.gz # in anywhere, maybe in your project directory
cd mmdetection-2.10.0
python setup.py develop

cd ..

# and move mmdet directory in mmdetection-2.10.0 to venv/lib/python3.7/site-packages
```

(10) Build TransFusion!

- **If you have type1**
  ``` shell
  git clone https://github.com/XuyangBai/TransFusion.git # in your project directory
  cd TransFusion
  python setup.py develop

  cd ..
  ```

- **If you have type2**

  The commit which 'Transfusion' forked from is 7c30072: "[Fixed] modify vote_head to support 3dssd (#396)".

  ***I merged the transfusion with the fourth commit(ID: d055876 / "update version file") right after it.***

  - It was possible to merge without affecting the implement of TransFusion for the following reasons.
  
      - The first commit right after: only changes the comments for the arguments.
      - The second commit right after: only changes them non-related with implement codes.
      - The third commit right after: fixes the problem between torch-1.8.0 and CUDA compiler(include 11.1).
      - The fourth commit right after: only changees version of version file.

  ``` shell
  git clone https://github.com/superego18/TransFusion-ChanJu.git # in your project directory
  cd TransFusion-ChanJu
  python setup.py develop
  
  cd ..
  ```

- Make path for mmdet3d in TransFusion directory
  ```shell
  cd ~ # go to the parent directory # example: chanju
  nano .bashrc # If you don't have the nano editor, use another editor.
  ```
  ``` bash
  # bash
  export PYTHONPATH=/home/{your_name}/{your_repo_name}/{TransFusion_repo_name}:/home/{your_name}/{your_repo_name}/venv/lib:$PYTHONPATH' # Add to end.
  ```
  ```shell
  source .bashrc
  activate
  ```

(12) Additional error fixes for **type2**.

- **If you have type 2**

    - Make pycocotools compatiable with mmdet. (When error is occured)
      - Download pycocotools again using a different method.
      ``` shell
      pip uninstall pycocotools
      pip install pycocotools==2.0.2 --no-cache-dir --no-binary :all:
      ```
      - Fix the assert version of pycocotools in mmdetection.
      ``` shell
      # venv/lib/python3.7/site-packages/mmdetection-2.10.0/mmdet/datasets/coco.py", line 21
      assert pycocotools.__version__ >= '2.0.2' # (12.0.2 --> 2.0.2)
      ```
      - Add version in pycocotools.
      ``` shell
      # venv/lib/python3.7/site-packages/pycocotools/init.py"
      __version__ = '2.0.2' (add on line 2)
      ```
      - Then you can run create_data.py
 
  - Train by nuscenes dataset
    - When you run train.py on nuscenes, maybe you will encounter the ouput "do not find nuscenes_gt_database ~~"
    - So you move only nuscenes_gt_database directory in data/nuscenes to your TransFusion directory.
    - I think the cause is probably due to the version difference of mmcv.
    - If you add print(filepath) to the def get(self, filepath) of fileio/file_client.py in mmcv, only the paths of files corresponding to nuscenes_gt_database come out differently. 

Presumably this will allow you to implement the demo code in getting_started.md and codes in data_preparation.md, train.py, etc.

### My environment info

When I run torch's collect_env code below

```shell
activate

python -m torch.utils.collect_env
```

- **My type 1 shows**
  ```shell
  PyTorch version: 1.6.0+cu101
  Is debug build: No
  CUDA used to build PyTorch: 10.1
  
  OS: Ubuntu 18.04.6 LTS
  GCC version: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
  CMake version: version 3.13.0
  
  Python version: 3.7
  Is CUDA available: Yes
  CUDA runtime version: 10.1.243
  GPU models and configuration: GPU 0: NVIDIA GeForce GTX 1660 Ti
  Nvidia driver version: 517.00
  cuDNN version: /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudnn.so.7.6.5
  
  Versions of relevant libraries:
  [pip3] numpy==1.19.5
  [pip3] torch==1.6.0+cu101
  [pip3] torchvision==0.7.0+cu101
  [conda] Could not collect
  ```
  
- **My type 2 shows**
  ```shell
  PyTorch version: 1.9.0+cu111
  Is debug build: False
  CUDA used to build PyTorch: 11.1
  ROCM used to build PyTorch: N/A
  
  OS: Ubuntu 18.04.6 LTS (x86_64)
  GCC version: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
  Clang version: Could not collect
  CMake version: version 3.13.0
  Libc version: glibc-2.26
  
  Python version: 3.7 (64-bit runtime)
  Python platform: Linux-5.15.133.1-microsoft-standard-WSL2-x86_64-with-Ubuntu-18.04-bionic
  Is CUDA available: True
  CUDA runtime version: 11.1.74
  GPU models and configuration: GPU 0: NVIDIA GeForce RTX 3080 Laptop GPU
  Nvidia driver version: 528.49
  cuDNN version: Probably one of the following:
  /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn.so.8.0.5
  /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.0.5
  /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.0.5
  /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.0.5
  /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.0.5
  /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.0.5
  /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.0.5
  HIP runtime version: N/A
  MIOpen runtime version: N/A
  
  Versions of relevant libraries:
  [pip3] numpy==1.19.5
  [pip3] torch==1.9.0+cu111
  [pip3] torchvision==0.10.0+cu111
  [conda] Could not collect
  ```

When I run train.py and I can see my env_info

- **My type 1 shows**
  ```shell
  sys.platform: linux
  Python: 3.7.5 (default, Dec  9 2021, 17:04:37) [GCC 8.4.0]
  CUDA available: True
  GPU 0: NVIDIA GeForce GTX 1660 Ti
  CUDA_HOME: /usr/local/cuda-10.1
  NVCC: Cuda compilation tools, release 10.1, V10.1.243
  GCC: gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
  PyTorch: 1.6.0+cu101
  PyTorch compiling details: PyTorch built with:
    - GCC 7.3
    - C++ Version: 201402
    - Intel(R) Math Kernel Library Version 2019.0.5 Product Build 20190808 for Intel(R) 64 architecture applications
    - Intel(R) MKL-DNN v1.5.0 (Git Hash e2ac1fac44c5078ca927cb9b90e1b3066a0b2ed0)
    - OpenMP 201511 (a.k.a. OpenMP 4.5)
    - NNPACK is enabled
    - CPU capability usage: AVX2
    - CUDA Runtime 10.1
    - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75
    - CuDNN 7.6.3
    - Magma 2.5.2
    - Build settings: BLAS=MKL, BUILD_TYPE=Release, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_VULKAN_WRAPPER -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, USE_CUDA=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_STATIC_DISPATCH=OFF,
  
  TorchVision: 0.7.0+cu101
  OpenCV: 4.9.0
  MMCV: 1.2.4
  MMCV Compiler: GCC 7.3
  MMCV CUDA Compiler: 10.1
  MMDetection: 2.10.0
  MMDetection3D: 0.11.0+
  ```

- **My type 2 shows**
  ```shell
  2024-01-11 05:53:50,299 - mmdet - INFO - Environment info:
  ------------------------------------------------------------
  sys.platform: linux
  Python: 3.7.5 (default, Dec  9 2021, 17:04:37) [GCC 8.4.0]
  CUDA available: True
  GPU 0: NVIDIA GeForce RTX 3080 Laptop GPU
  CUDA_HOME: /usr/local/cuda-11.1
  NVCC: Build cuda_11.1.TC455_06.29069683_0
  GCC: gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
  PyTorch: 1.9.0+cu111
  PyTorch compiling details: PyTorch built with:
    - GCC 7.3
    - C++ Version: 201402
    - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
    - Intel(R) MKL-DNN v2.1.2 (Git Hash 98be7e8afa711dc9b66c8ff3504129cb82013cdb)
    - OpenMP 201511 (a.k.a. OpenMP 4.5)
    - NNPACK is enabled
    - CPU capability usage: AVX2
    - CUDA Runtime 11.1
    - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
    - CuDNN 8.0.5
    - Magma 2.5.2
    - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.1, CUDNN_VERSION=8.0.5, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON,
  
  TorchVision: 0.10.0+cu111
  OpenCV: 4.9.0
  MMCV: 1.3.10
  MMCV Compiler: GCC 7.3
  MMCV CUDA Compiler: 11.1
  MMDetection: 2.10.0
  MMDetection3D: 0.12.0+e8cc677
  ```  

# TransFusion repository

PyTorch implementation of TransFusion for CVPR'2022 paper ["TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers"](https://arxiv.org/abs/2203.11496), by Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu and Chiew-Lan Tai.

This paper focus on LiDAR-camera fusion for 3D object detection. If you find this project useful, please cite:

```bash
@article{bai2021pointdsc,
  title={{TransFusion}: {R}obust {L}iDAR-{C}amera {F}usion for {3}D {O}bject {D}etection with {T}ransformers},
  author={Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu and Chiew-Lan Tai},
  journal={CVPR},
  year={2022}
}
```

## Introduction

LiDAR and camera are two important sensors for 3D object detection in autonomous driving. Despite the increasing popularity of sensor fusion in this field, the robustness against inferior image conditions, e.g., bad illumination and sensor misalignment, is under-explored. Existing fusion methods are easily affected by such conditions, mainly due to a hard association of LiDAR points and image pixels, established by calibration matrices.
We propose TransFusion, a robust solution to LiDAR-camera fusion with a soft-association mechanism to handle inferior image conditions. Specifically, our TransFusion consists of convolutional backbones and a detection head based on a transformer decoder. The first layer of the decoder predicts initial bounding boxes from a LiDAR point cloud using a sparse set of object queries, and its second decoder layer adaptively fuses the object queries with useful image features, leveraging both spatial and contextual relationships. The attention mechanism of the transformer enables our model to adaptively determine where and what information should be taken from the image, leading to a robust and effective fusion strategy. We additionally design an image-guided query initialization strategy to deal with objects that are difficult to detect in point clouds. TransFusion achieves state-of-the-art performance on large-scale datasets. We provide extensive experiments to demonstrate its robustness against degenerated image quality and calibration errors. We also extend the proposed method to the 3D tracking task and achieve the 1st place in the leaderboard of nuScenes tracking, showing its effectiveness and generalization capability.

![pipeline](resources/pipeline.png)

**updates**
- March 23, 2022: paper link added
- March 15, 2022: initial release

## Main Results

Detailed results can be found in [nuscenes.md](configs/nuscenes.md) and [waymo.md](configs/waymo.md). Configuration files and guidance to reproduce these results are all included in [configs](configs), we are not going to release the pretrained models due to the policy of Huawei IAS BU. 

### nuScenes detection test 

| Model   | Backbone | mAP | NDS  | Link  |
|---------|--------|--------|---------|---------|
| [TransFusion-L](configs/transfusion_nusc_voxel_L.py) | VoxelNet | 65.52 | 70.23 | [Detection](https://drive.google.com/file/d/1Wk8p2LJEhwfKfhsKzlU9vDBOd0zn38dN/view?usp=sharing)
| [TransFusion](configs/transfusion_nusc_voxel_LC.py) | VoxelNet | 68.90 | 71.68 | [Detection](https://drive.google.com/file/d/1X7_ig4v5A2vKsiHtUGtgeMN-0RJKsM6W/view?usp=sharing)

### nuScenes tracking test

| Model | Backbone | AMOTA |  AMOTP   | Link  |
|---------|--------|--------|---------|---------|
| [TransFusion-L](configs/transfusion_nusc_voxel_L.py) | VoxelNet | 0.686 | 0.529 | [Detection](https://drive.google.com/file/d/1Wk8p2LJEhwfKfhsKzlU9vDBOd0zn38dN/view?usp=sharing) / [Tracking](https://drive.google.com/file/d/1pKvRBUsM9h1Xgturd0Ae_bnGt0m_j3hk/view?usp=sharing)| 
| [TransFusion](configs/transfusion_nusc_voxel_LC.py)| VoxelNet | 0.718 | 0.551 | [Detection](https://drive.google.com/file/d/1X7_ig4v5A2vKsiHtUGtgeMN-0RJKsM6W/view?usp=sharing) / [Tracking](https://drive.google.com/file/d/1EVuS-MAg_HSXUVqMrXEs4-RpZp0p5cfv/view?usp=sharing)| 

### waymo detection validation

| Model   | Backbone | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   |
|---------|--------|---------|---------|---------|---------|
| [TransFusion-L](configs/transfusion_waymo_voxel_L.py) | VoxelNet | 65.07 | 63.70 | 65.97 | 64.91
| [TransFusion](configs/transfusion_waymo_voxel_LC.py) | VoxelNet | 65.11 | 64.02 | 67.40 | 65.51

## Use TransFusion

**Installation**

Please refer to [getting_started.md](docs/getting_started.md) for installation of mmdet3d. We use mmdet 2.10.0 and mmcv 1.2.4 for this project.

**Benchmark Evaluation and Training**

Please refer to [data_preparation.md](docs/data_preparation.md) to prepare the data. Then follow the instruction there to train our model. All detection configurations are included in [configs](configs/). 

Note that if you a the newer version of mmdet3d to prepare the meta file for nuScenes and then train/eval the TransFusion, it will have a wrong mAOE and mASE because mmdet3d has a [coordinate system refactoring](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/compatibility.md#coordinate-system-refactoring) which affect the definitation of yaw angle and object size (`l, w`).

## Acknowlegement

We sincerely thank the authors of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [CenterPoint](https://github.com/tianweiy/CenterPoint), [GroupFree3D](https://github.com/zeliu98/Group-Free-3D) for open sourcing their methods.
