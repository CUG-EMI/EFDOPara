# EFDOPara
🚀 Stop waiting, start training! Slash training times for geophysical neural operators on multi-GPU setups with our dynamic training strategy, without sacrificing accuracy.
## 🛠️ Installation

> 💡 Pro tip: We recommend using `Anaconda` with `Mamba` for lightning-fast package installation!

### Step 1: Get Mamba Up and Running

First, grab Mamba from the [Mambaforge download page](https://github.com/conda-forge/miniforge#miniforge):

```bash
bash Miniforge3-Linux-x86_64.sh -b -p ${HOME}/miniforge
```

### Step 2: Set Up Your Environment

Add these magic lines to your `~/.bashrc`:

```bash
# conda
if [ -f "${HOME}/miniforge/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniforge/etc/profile.d/conda.sh"
fi
# mamba
if [ -f "${HOME}/miniforge/etc/profile.d/mamba.sh" ]; then
    source "${HOME}/miniforge/etc/profile.d/mamba.sh"
fi

alias conda=mamba
```

### Step 3: Create Your EFDO Environment

```bash
conda create -n torch python=3.11
conda activate torch
```

### Step 4: Install Dependencies

```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install other required packages
conda install torchinfo pyyaml numpy scipy pandas matplotlib jupyter notebook
pip install ray
```

### Step 5: Get the Code

```bash
git clone https://github.com/CUG-EMI/EFDOPara
```