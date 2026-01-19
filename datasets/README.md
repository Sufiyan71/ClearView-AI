# ClearView-AI - CycleGAN Setup Guide

A comprehensive guide for setting up and training CycleGAN for image dehazing on Windows and macOS.

### Clone the Repository
```bash
git clone https://github.com/Sufiyan71/ClearView-AI.git
cd ClearView-AI
```

---

## Installation Methods

### Method 1: Conda/Miniconda

#### Install Conda/Miniconda

**Windows:**
1. Download Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Run the installer and follow the prompts
3. Open "Anaconda Prompt" from Start Menu

**macOS:**
```bash
# Download and install Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
# Follow the prompts, then restart terminal
```

#### Create and Activate Environment

```bash
# Create conda environment
conda create -n cyclegan python=3.11 -y

# Activate environment
conda activate cyclegan

# Install PyTorch with CUDA (Windows/Linux with NVIDIA GPU)
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install PyTorch for CPU only (Windows/macOS without GPU)
conda install pytorch==2.4.0 torchvision==0.19.0 cpuonly -c pytorch -y

# Install PyTorch for macOS with MPS (Apple Silicon)
conda install pytorch==2.4.0 torchvision==0.19.0 -c pytorch -y

# Install other dependencies
conda install -c conda-forge numpy=1.24.3 scikit-image pip -y
pip install Pillow==10.0.0 dominate==2.8.0 wandb==0.16.0
```

---

### Method 2: Python Virtual Environment (venv)

#### Windows

```bash
# Navigate to project directory
cd ClearView-AI

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support (NVIDIA GPU)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# OR Install PyTorch CPU-only version
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install numpy==1.24.3 scikit-image==0.21.0 Pillow==10.0.0 dominate==2.8.0 wandb==0.16.0
```

#### macOS

```bash
# Navigate to project directory
cd ClearView-AI

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (Apple Silicon M1/M2 with MPS support)
pip install torch==2.4.0 torchvision==0.19.0

# Install remaining dependencies
pip install numpy==1.24.3 scikit-image==0.21.0 Pillow==10.0.0 dominate==2.8.0 wandb==0.16.0
```

---

### Method 3: Using YAML Environment File

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate environment
conda activate cyclegan
```

---

## Dataset Preparation

### Directory Structure

Your dataset should follow this structure:

```
datasets/
└── your_dataset_name/
    ├── trainA/          # Domain A training images (e.g., hazy images)
    ├── trainB/          # Domain B training images (e.g., clear images)
    ├── testA/           # Domain A test images
    └── testB/           # Domain B test images
```

### Data Split Guidelines

- **Training Set**: 90% of your data (trainA and trainB)
- **Test Set**: 10% of your data (testA and testB)
- Supported formats: `.png`, `.jpg`, `.jpeg`
```

---

## Training

### Standard Resolution (256x256)

**Recommended for most GPUs with 4GB+ VRAM**

```bash
python train.py \
    --dataroot ./datasets/your_dataset_name \
    --name dehaze_cyclegan_256 \
    --model cycle_gan \
    --preprocess resize \
    --load_size 256 \
    --crop_size 256 \
    --batch_size 1 \
    --gpu_ids 0
```

**Windows (PowerShell):**
```powershell
python train.py --dataroot ./datasets/your_dataset_name --name dehaze_cyclegan_256 --model cycle_gan --preprocess resize --load_size 256 --crop_size 256 
```

### Medium Resolution (356x356)

**For GPUs with 6GB+ VRAM**

```bash
python train.py --dataroot ./datasets/your_dataset_name --name dehaze_cyclegan_256 --model cycle_gan --preprocess resize --load_size 356 --crop_size 356 
```

### High Resolution (512x512)

**For high-end GPUs with 8GB+ VRAM (RTX 3080, A100, etc.)**

```bash
python train.py --dataroot ./datasets/your_dataset_name --name dehaze_cyclegan_256 --model cycle_gan --preprocess resize --load_size 512 --crop_size 512 
```

### CPU Training (No GPU)

```bash
python train.py \
    --dataroot ./datasets/your_dataset_name \
    --name dehaze_cyclegan_cpu \
    --model cycle_gan \
    --preprocess resize \
    --load_size 256 \
    --crop_size 256 \
    --gpu_ids -1
```

### Additional Training Options

- `--display_id -1`: Disable Visdom display
- `--save_epoch_freq 5`: Save model every 5 epochs
- `--print_freq 100`: Print losses every 100 iterations
- `--continue_train`: Continue training from checkpoint
- `--epoch_count 1`: Starting epoch count

---

## Testing

### Test with 256x256 Resolution

```bash
python test.py \
    --dataroot ./datasets/your_dataset_name \
    --name dehaze_cyclegan_256 \
    --model cycle_gan \
    --preprocess resize \
    --load_size 256 \
    --crop_size 256 \
    --gpu_ids 0
```

### Test with 356x356 Resolution

```bash
python test.py \
    --dataroot ./datasets/your_dataset_name \
    --name dehaze_cyclegan_356 \
    --model cycle_gan \
    --preprocess resize \
    --load_size 356 \
    --crop_size 356 \
    --gpu_ids 0
```

### Test with 512x512 Resolution

```bash
python test.py \
    --dataroot ./datasets/your_dataset_name \
    --name dehaze_cyclegan_512 \
    --model cycle_gan \
    --preprocess resize \
    --load_size 512 \
    --crop_size 512 \
    --gpu_ids 0
```

### CPU Testing

```bash
python test.py \
    --dataroot ./datasets/your_dataset_name \
    --name dehaze_cyclegan_256 \
    --model cycle_gan \
    --preprocess resize \
    --load_size 256 \
    --crop_size 256 \
    --gpu_ids -1
```

### Test Results

Results will be saved to: `./results/[name]/test_latest/`

---

## Training

### Output and Checkpoints

During training, all model checkpoints and training logs will be saved in the `checkpoints` folder:

```
checkpoints/
└── [your_model_name]/
    ├── latest_net_G_A.pth      # Generator A checkpoint
    ├── latest_net_G_B.pth      # Generator B checkpoint
    ├── latest_net_D_A.pth      # Discriminator A checkpoint
    ├── latest_net_D_B.pth      # Discriminator B checkpoint
    ├── loss_log.txt            # Training loss logs
    ├── train_opt.txt           # Training options/parameters
    └── web/                    # HTML visualization files
        └── images/             # Training progress images
```
