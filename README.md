# Dense PPO for Sequence Generation

**NeurIPS 2025 Anonymous Submission**

Implementation of "Adversarial Learning for Sequence Generation: A GAIL-Based Approach"

## Quick Start

```bash
# Clone and setup
git clone [anonymous-repo]
cd dPPO-NeurIPS
pip install -r requirements.txt

# Run single experiment
python run_all_experiments.py --model dppo --experiment vwap --seed 0

# Run all paper experiments (takes 12-24 hours)
python run_all_experiments.py --model all --experiment all --num_seeds 30
```

## Repository Structure

```
dPPO-NeurIPS/
├── dPPO/                    # Dense reward experiments
│   ├── arma-garch/         # ARMA-GARCH dataset (10→100 steps)
│   ├── electricity/        # Electricity dataset (20→200 steps)
│   ├── oracle/             # Oracle experiments
│   └── vwap/               # VWAP dataset (75→350 steps)
├── sPPO/                    # Sparse reward experiments  
│   ├── arma-garch/
│   ├── electricity/
│   ├── oracle/
│   └── vwap/
├── SeqGAN/                  # Original SeqGAN baseline
│   └── oracle/             # Only oracle experiments
└── run_all_experiments.py  # Simple experiment runner
```

## Models

- **dPPO**: Our dense reward method (token-level feedback)
- **sPPO**: Sparse reward baseline (sequence-level feedback)  
- **SeqGAN**: Original SeqGAN with REINFORCE (oracle only)

## Individual Experiments

```bash
# Dense PPO experiments
python run_all_experiments.py --model dppo --experiment vwap --seed 0
python run_all_experiments.py --model dppo --experiment electricity --seed 0
python run_all_experiments.py --model dppo --experiment arma-garch --seed 0
python run_all_experiments.py --model dppo --experiment oracle --seed 0

# Sparse PPO experiments
python run_all_experiments.py --model sppo --experiment vwap --seed 0

# SeqGAN baseline (oracle only)
python run_all_experiments.py --model seqgan --experiment oracle --seed 0
```

## Reproduce Paper Results

### Table 1 Results (VWAP, Electricity, ARMA-GARCH)
```bash
# Run all methods on all datasets with 30 seeds
python run_all_experiments.py --model all --experiment all --num_seeds 30
```

### Figure 2 Results (Long sequence extrapolation)
The same command above generates the extrapolation curves shown in Figure 2.

### Oracle Experiments (Appendix)
```bash
# Oracle experiments including SeqGAN
python run_all_experiments.py --model all --experiment oracle --num_seeds 30
```

## Expected Results

**VWAP Dataset (Table 1):**
| Method | JS Div ↓ | KL Div ↓ | ACF MSE ↓ |
|--------|----------|----------|-----------|
| sPPO | 0.0546±0.0283 | 0.0160±0.0170 | 0.000478±0.000011 |
| **dPPO** | **0.0422±0.0139** | **0.0085±0.0056** | **0.000477±0.000009** |

Results saved to `results/` directory in each experiment folder.

## Requirements

- Python 3.8+
- PyTorch 2.1+
- CUDA GPU recommended (8GB+ VRAM)
- See `requirements.txt` for full dependencies

## File Organization

Each experiment folder contains:
- `train.py` - Main training script
- `train_parallel.py` - Parallel hyperparameter search
- `generator.py` - Generator model implementation
- `discriminator.py` - Discriminator model implementation
- `environment.py` - RL environment for token generation
- `callback.py` - Training callbacks and evaluation
- `data/` - Dataset files (.npy format)
- `saved_models/` - Pretrained model checkpoints

## Hardware Requirements

- **Recommended**: NVIDIA GPU with 8GB+ VRAM
- **Minimum**: 4GB GPU memory 
- **RAM**: 16GB+ system memory
- **Time**: Full reproduction takes 12-24 hours on modern GPUs