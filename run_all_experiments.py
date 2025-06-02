#!/usr/bin/env python3
"""
Simple runner for NeurIPS 2025 dPPO experiments.

Usage:
    python run_all_experiments.py --model dppo --experiment vwap --seed 0
    python run_all_experiments.py --model all --experiment all --num_seeds 5
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_experiment(model_type, experiment, seed=0, gpu=0):
    """Run a single experiment by calling the train.py in the appropriate folder."""
    
    repo_root = Path(__file__).parent
    
    # Determine the folder path
    if model_type == 'dppo':
        folder_path = repo_root / "dPPO" / experiment
    elif model_type == 'sppo':
        folder_path = repo_root / "sPPO" / experiment  
    elif model_type == 'seqgan':
        if experiment != 'oracle':
            print("SeqGAN only available for oracle experiment")
            return False
        folder_path = repo_root / "SeqGAN" / "oracle"
    else:
        print("Unknown model type: {}".format(model_type))
        return False
    
    if not folder_path.exists():
        print("Folder not found: {}".format(folder_path))
        return False
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': str(gpu),
        'SEED': str(seed),
        'PYTHONPATH': str(folder_path)
    })
    
    # Run the training script in that folder
    train_script = folder_path / "train_parallel.py"
    if not train_script.exists():
        print("train_parallel.py not found in {}".format(folder_path))
        return False
    
    print("Running {} {} seed {} in {}".format(model_type, experiment, seed, folder_path))
    
    # Set environment variables for the parallel script
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': str(gpu),
        'SEED': str(seed),
        'PYTHONPATH': str(folder_path)
    })
    
    # Run the parallel training script in that folder
    result = subprocess.run(
        [sys.executable, "train_parallel.py"], 
        cwd=folder_path,
        env=env,
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Completed {} {} seed {}".format(model_type, experiment, seed))
        return True
    else:
        print("✗ Failed {} {} seed {}".format(model_type, experiment, seed))
        print("Error: {}...".format(result.stderr[:500]))  # Show first 500 chars of error
        return False

def main():
    parser = argparse.ArgumentParser(description='Run dPPO experiments')
    parser.add_argument('--model', 
                       choices=['dppo', 'sppo', 'seqgan', 'all'], 
                       required=True, 
                       help='Model type to run')
    parser.add_argument('--experiment', 
                       choices=['vwap', 'electricity', 'arma-garch', 'oracle', 'all'],
                       required=True, 
                       help='Experiment to run')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_seeds', type=int, default=30, help='Number of seeds for "all" mode')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Determine which models to run
    if args.model == 'all':
        models = ['dppo', 'sppo']
        if args.experiment in ['oracle', 'all']:
            models.append('seqgan')
    else:
        models = [args.model]
    
    # Determine which experiments to run
    if args.experiment == 'all':
        experiments = ['vwap', 'electricity', 'arma-garch', 'oracle']
    else:
        experiments = [args.experiment]
    
    # Run experiments
    total_runs = 0
    successful_runs = 0
    
    for model in models:
        for exp in experiments:
            # Skip invalid combinations
            if model == 'seqgan' and exp != 'oracle':
                continue
                
            if args.model == 'all' and args.experiment == 'all':
                # Multiple seeds
                for seed in range(args.num_seeds):
                    total_runs += 1
                    if run_experiment(model, exp, seed, args.gpu):
                        successful_runs += 1
            else:
                # Single run
                total_runs += 1
                if run_experiment(model, exp, args.seed, args.gpu):
                    successful_runs += 1
    
    print("\nCompleted {}/{} experiments".format(successful_runs, total_runs))
    if successful_runs == total_runs:
        print("🎉 All experiments completed successfully!")
    else:
        print("⚠️  {} experiments failed".format(total_runs - successful_runs))

if __name__ == '__main__':
    main()