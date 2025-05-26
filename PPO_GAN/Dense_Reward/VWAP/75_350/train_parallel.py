import os
import torch as th
import subprocess
from pathlib import Path
import json
import numpy as np
import itertools
import time
import hashlib
import pickle

# ============= BASE DIRECTORIES =============
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results_vwap"
METRICS_DIR = BASE_DIR / "saved_metrics_training"
MODELS_DIR = BASE_DIR / "saved_models_training"

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============= PARALLEL TRAINING PARAMETERS =============
# Settings for hyperparameter search
PARALLEL_CONFIG = {
    
    'num_seeds': 30,
    
    'param_grid': {

        # Generator architecture
        'g_hidden_dim': [256],
        
        # Evaluation
        'eval_freq': [1],

        # Pretraining Generator
        'g_pretrain_lr': [5e-4],
        'g_pretrain_batch_size': [128],
        'g_pretrain_epochs': [100],
        'g_eval_pretrain_epochs': [5],

        # Discriminator parameters
        'd_pretrain_lr': [1e-4],
        'd_batch_size': [128],
        'd_learning_rate': [1e-7],
        
        # Number of batches for discriminator training per PPO rollout
        'd_epochs': [10],

        # PPO parameters
        'ppo_total_timesteps': [75 * 80],
        'ppo_n_steps': [75 * 2],
        'ppo_batch_size': [75 * 1],
        'ppo_n_epochs': [2],

        'use_linear_lr_decay': [False],
        'min_ppo_lr': [1e-5],

        'ppo_learning_rate': [9e-6],
        
        'ppo_gamma': [0.999],
        'ppo_gae_lambda': [0.95],
        'ppo_clip_range': [0.1],
        'ppo_ent_coef': [0.001],
        'ppo_vf_coef': [0.75],
        'ppo_clip_range_vf': [None],
        'ppo_max_grad_norm': [0.5],
        
        'do_pretrain': [False],
        'do_hyperparam_search': [False],
        
        # Weight transfer
        'transfer_weights': [True],
        'transfer_head': [True]
    },
    'output_dir': RESULTS_DIR,
}


def get_config_hash(config):
    """Generate a unique hash for a configuration."""
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]

def get_free_gpus():
    """Find all free GPUs to use from the allowed GPUs."""
    allowed_gpus = [0,3,4,5,6,7]  # Only use these GPUs
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,nounits,noheader'], 
            capture_output=True, text=True, timeout=3
        )
        
        if result.returncode != 0:
            return allowed_gpus  # Default to all allowed GPUs if check fails
        
        free_gpus = []
        
        for i, line in enumerate(result.stdout.strip().split('\n')):
            if i in allowed_gpus:  # Only check allowed GPUs
                used, free, util = map(int, line.split(','))
                if used < 100 and util < 5:  # Consider GPU free if low usage
                    free_gpus.append(i)
                
        return free_gpus if free_gpus else allowed_gpus
    except Exception as e:
        print(f"Error checking GPU status: {e}")
        return allowed_gpus  # Default to all allowed GPUs

def generate_configs(param_grid):
    """Generate all possible configurations from the parameter grid."""
    keys = param_grid.keys()
    values = param_grid.values()
    return [dict(zip(keys, v)) for v in itertools.product(*values)]    

def run_training(config, gpu_id, seed, output_dir, config_id):
    """
    Run a single training job as a subprocess.
    
    Args:
        config: Configuration dictionary
        gpu_id: GPU ID to use
        seed: Random seed value
        output_dir: Directory to save results
    
    Returns:
        subprocess.Popen: Running process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration to file
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Prepare environment variables
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "CONFIG_PATH": str(config_path),
        "OUTPUT_DIR": str(output_dir),
        "SEED": str(seed),
        "CONFIG_ID": str(config_id),  # Add this line
        "WORKING_DIR": str(BASE_DIR),
        "METRICS_DIR": str(BASE_DIR / "saved_metrics_training"),
        "MODELS_DIR": str(BASE_DIR / "saved_models_training")     
        })

    # Start training process
    process = subprocess.Popen(
        ["python3", str(BASE_DIR / "train.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def main():
    """Main function to run parallel training."""
    # Create output directory
    output_dir = PARALLEL_CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all configurations
    configs = generate_configs(PARALLEL_CONFIG['param_grid'])
    
    # Save configurations
    with open(output_dir / "all_configs.json", 'w') as f:
        json.dump(configs, f, indent=2)
    
    print(f"Generated {len(configs)} configurations")
    print(f"Running {PARALLEL_CONFIG['num_seeds']} seeds for each configuration")
    print(f"Total runs: {len(configs) * PARALLEL_CONFIG['num_seeds']}")
    
    # Track active processes
    active_processes = {}  # (config_id, seed): {'process': process, 'gpu': gpu_id, 'output_dir': output_dir}
    completed_runs = set()  # Set of (config_id, seed) that have completed

    while len(completed_runs) < len(configs) * PARALLEL_CONFIG['num_seeds']:
        # Get all available GPUs
        free_gpus = get_free_gpus()
        
        # Start a job on each free GPU if we have pending runs
        for gpu_id in free_gpus:
            # Skip if this GPU already has a process running
            if any(info['gpu'] == gpu_id for info in active_processes.values()):
                continue
                
            # Find a configuration and seed to run
            found_job = False
            for config_id, config in enumerate(configs):
                for seed in range(PARALLEL_CONFIG['num_seeds']):
                    run_key = (config_id, seed)
                    
                    if run_key not in completed_runs and run_key not in active_processes:
                        # Create output directory for this run
                        run_dir = output_dir / f"config_{config_id}_seed_{seed}"
                        
                        # Start the training process
                        try:
                            print(f"Starting config {config_id}, seed {seed} on GPU {gpu_id}")
                            process = run_training(config, gpu_id, seed, run_dir, config_id) 
                            
                            # Track the process
                            active_processes[run_key] = {
                                'process': process,
                                'gpu': gpu_id,
                                'output_dir': run_dir,
                                'start_time': time.time()
                            }
                            
                            found_job = True
                            break
                        except Exception as e:
                            print(f"Error starting run for config {config_id}, seed {seed}: {e}")
                            continue
                            
                if found_job:
                    break
        
        # Check active processes for completion
        for run_key in list(active_processes.keys()):
            process_info = active_processes[run_key]
            process = process_info['process']
            
            # Check if process has completed
            if process.poll() is not None:
                config_id, seed = run_key
                run_dir = process_info['output_dir']
                
                # Get output from process
                stdout, stderr = process.communicate()
                
                # Save logs
                with open(run_dir / "stdout.log", 'w') as f:
                    f.write(stdout)
                with open(run_dir / "stderr.log", 'w') as f:
                    f.write(stderr)
                
                # Check if successful
                if process.returncode == 0:
                    elapsed = time.time() - process_info['start_time']
                    print(f"Completed config {config_id}, seed {seed} in {elapsed:.1f} seconds")
                    completed_runs.add(run_key)
                else:
                    print(f"Failed config {config_id}, seed {seed}, return code: {process.returncode}")
                    # Still mark as completed to avoid retrying
                    completed_runs.add(run_key)
                
                # Remove from active processes
                del active_processes[run_key]
        
        # Print status update periodically
        if len(completed_runs) % 5 == 0 and len(completed_runs) > 0:
            print(f"Progress: {len(completed_runs)}/{len(configs) * PARALLEL_CONFIG['num_seeds']} runs completed")
            print(f"Active processes: {len(active_processes)}")
        
        # Sleep to prevent CPU spinning
        time.sleep(4)
    
    print("\nAll training runs completed!")
    
    print(f"All done! Results saved to {output_dir}")

if __name__ == "__main__":
    main()