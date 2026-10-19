# Anonymous Submission

This repository contains the official anonymous code and data for our submission to **[NeurIPS, 2026]**.

## 📋 Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [License](#license)


## Requirements

- Python 3.8.20
- PyTorch 2.0.0+cu118
- CUDA 11.8
- PyTorch Geometric 2.0.4
Tested on: Ubuntu 22.04.5 LTS, NVIDIA GeForce RTX 4090 with 24 GB memory.

## Installation

```bash
# Create environment
conda create -n project python=3.8.20 -y
conda activate project

# Install dependencies
pip install -r requirements.txt
```


## Dataset

The dataset is hosted anonymously and can be downloaded from:

🔗 **Anonymous Dataset Link:** [https://osf.io/dnrb7/overview?view_only=c71562be99ac4ca782e8f315b9d8b3fe](#)


## Quick Start

```bash
python main.py --dataset collab
python main.py --dataset act
python main.py --dataset synthetic --P 0.4
python main.py --dataset synthetic --P 0.6
python main.py --dataset synthetic --P 0.8
python main.py --dataset dymotif_data
python main.py --dataset Aminer

```

### Supported Datasets

**Link Prediction**
- `collab`
- `act`
- `synthetic` (0.4, 0.6, 0.8)

**Node Classification**
- `Aminer`
- `dymotif_data`

## License

This code is released for **anonymous review purposes only**. 
