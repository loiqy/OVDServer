# External Dependencies

This directory contains external repositories managed as Git submodules.

## How to Initialize Submodules

1. Clone the main repository:
    ```bash
    git clone https://github.com/yourusername/yourrepo.git
    ```

2. Navigate to the repository:
    ```bash
    cd yourrepo
    ```

3. Initialize and update the submodules:
    ```bash
    git submodule init
    git submodule update
    ```

## List of Submodules

- `APE`: A submodule for [APE](https://github.com/shenyunhang/APE).
    - Path: `ovdserver/backends/APE/shenyunhang/APE`
    - Branch: `main`

## Updating Submodules

To update a submodule to the latest commit:

1. Navigate to the submodule directory:
    ```bash
    cd ovdserver/backends/APE/shenyunhang/APE
    ```

2. Pull the latest changes:
    ```bash
    git pull
    ```

3. Return to the main repository and commit the updated submodule:
    ```bash
    cd ../../../../
    git add ovdserver/backends/APE/shenyunhang/APE
    git commit -m "Update APE submodule"
    ```

## Install APE

0. 新建虚拟环境，并安装torch2.5.1cu121

    ```bash
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    ```

1. 修改APE/requirements.txt

    将三个仓库的commit设置为新版（旧版安装有很多问题），并且注释torch以及指定版本了的transformers。[requirements.txt](ape_latest_requirements.txt)如下：

    ```
    #torch==1.12.1
    #torchvision
    #transformers==4.32.1
    transformers
    cython
    opencv-python
    opencv-python-headless
    scipy
    einops
    lvis
    fairscale
    git+https://github.com/facebookresearch/detectron2@c69939aa85460e8135f40bce908a6cddaa73065f
    git+https://github.com/IDEA-Research/detrex@c56d32e3d0262cff9835ebe80a0642965ae0cb3e
    git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
    ```

2. 运行安装

    首先要确保nvcc正确（/usr/local/cuda-12.1），并且与pytorch的cuda版本(12.1)匹配，否则会报错。

    1) 检查nvcc版本

    ```bash
    nvcc --version
    ```

    这将显示系统当前使用的CUDA编译器版本。如果显示的是11.8，那么你需要确保系统使用的是CUDA 12.1。

    2) 确保环境变量PATH和LD_LIBRARY_PATH指向正确的CUDA版本

    ```bash
    export PATH=/usr/local/cuda-12.1/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
    ```

    3) (optional) xformers

    安装与torch2.5.1cu121匹配的xformers: 0.0.28.post3

    ```bash
    pip3 install -U xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121
    ```

    安装完成后，使用环境变量激活使用APE的OVDServer的xformers的使用

    ```bash
    export OVDSERVER_USE_XFORMERS=1
    ```

    4) 安装APE

    ```bash
    pip3 install -r requirements.txt
    cd shenyunhang/APE
    pip install -e .
    ```
