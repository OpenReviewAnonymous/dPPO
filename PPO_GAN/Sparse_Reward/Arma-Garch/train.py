import os
import time
import json
import random
import numpy as np
import torch as th
from pathlib import Path
import sys
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn
from sb3_contrib.ppo_recurrent import RecurrentPPO
from scipy.stats import wasserstein_distance
from scipy.special import kl_div

# Import local modules
from generator import Generator, pretrain_generator, transfer_weights_from_saved, generate_ppo_samples, calculate_ppo_metrics, evaluate_best_model
from discriminator import Discriminator, pretrain_discriminator
from environment import TokenGenerationEnv
from callback import CustomCallback

# ============= BASE DIRECTORIES =============
BASE_DIR = Path(os.getenv('WORKING_DIR', Path(os.path.dirname(os.path.abspath(__file__)))))
SAVE_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results_electricity"
METRICS_DIR = BASE_DIR / "saved_metrics_training"
MODELS_DIR = BASE_DIR / "saved_models_training"
DATA_DIR = BASE_DIR / "data"  # Directory for electricity data

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============= FIXED PARAMETERS ============= #
# VOCABULARY SIZE               ==============
# TRAIN LENGTH                  ==============
# INFERENCE LENGTH              ==============
# DATA FILESET                  ==============
# ============= FIXED PARAMETERS ============= #


# ============= FIXED PARAMETERS =============
# Data parameters
VOCAB_SIZE = 200  
TRAIN_SEQ_LENGTH = 10           # Short sequence for training
INFERENCE_SEQ_LENGTH = 100      # Long sequence for inference
# ============= FIXED PARAMETERS ============= #

# GENERATOR
G_NUM_LAYERS = 2
G_LR_PATIENCE = 5
G_LR_DECAY = 0.5

# DISCRIMINATOR
DISCRIMINATOR_EMB_DIM = 128
DISCRIMINATOR_HIDDEN_DIM = 256
D_DROPOUT_RATE = 0.0
D_PRETRAIN_EPOCHS = 20
D_LR_PATIENCE = 10
D_LR_DECAY = 0.7
D_LR_MIN = 1e-5

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def load_data():
    """Load and return electricity sequence data."""
    print("Loading electricity sequence data...")
    
    train_data = np.load(DATA_DIR / 'train_vwap_10_200.npy')
    val_data_short = np.load(DATA_DIR / 'val_short_vwap_10_200.npy')
    val_data_long = np.load(DATA_DIR / 'val_long_vwap_100_200.npy')
    test_data = np.load(DATA_DIR / 'test_vwap_100_200.npy')
    
    print(f"Loaded data:")
    print(f"  Train: {train_data.shape}")
    print(f"  Validation (short): {val_data_short.shape}")
    print(f"  Validation (long): {val_data_long.shape}")
    print(f"  Test: {test_data.shape}")
    
    return train_data, val_data_short, val_data_long, test_data

def create_start_token_distribution(train_data):
    """Create a distribution for start tokens from training data"""
    # Flatten the training data to get all tokens
    all_tokens = train_data.flatten()
    
    # Create a histogram (count) for each token in the vocabulary
    token_counts = np.bincount(all_tokens, minlength=VOCAB_SIZE)
    
    # Convert to probability distribution
    token_probs = token_counts / token_counts.sum()
    
    return token_probs

def main():
    """Main function to run a single PPO-SeqGAN training."""
    
    # Get environment variables
    config_path = os.getenv('CONFIG_PATH')
    seed = int(os.getenv('SEED', '0'))

    output_dir = Path(os.getenv('OUTPUT_DIR', RESULTS_DIR / "elcitricity_ppo"))
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Load configuration
    print(f"Loading config from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if th.cuda.is_available():
        gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        device = th.device("cuda:0")
        print(f"Using GPU: {gpu_id}")
    else:
        device = th.device("cpu")
        print("Using CPU")
    
    # Start timing
    start_time = time.time()
    
    if config.get('do_hyperparam_search', False):
        config_id = os.getenv('CONFIG_ID', '0')
        gen_pretrain_log = os.path.join(METRICS_DIR, f"config_{config_id}_seed_{seed}_generator_pretrain.txt")
        disc_pretrain_log = os.path.join(METRICS_DIR, f"config_{config_id}_seed_{seed}_discriminator_pretrain.txt")
        ppo_log = os.path.join(METRICS_DIR, f"config_{config_id}_seed_{seed}_adversarial_train.txt")
    else:
        gen_pretrain_log = os.path.join(METRICS_DIR, f"{seed}_generator_pretrain.txt")
        disc_pretrain_log = os.path.join(METRICS_DIR, f"{seed}_discriminator_pretrain.txt")
        ppo_log = os.path.join(METRICS_DIR, f"{seed}_adversarial_train.txt")
    
    # Print training configuration
    print(f"Training DSGAN with:")
    print(f"  Seed: {seed}")
    print(f"  Device: {device}")
    print(f"  Generator Hidden Dim: {config['g_hidden_dim']}")
    print(f"  PPO Learning Rate: {config['ppo_learning_rate']}")
    print(f"  Discriminator Learning Rate: {config['d_learning_rate']}")
    print(f"  PPO Total Timesteps: {config['ppo_total_timesteps']}")
    
    # Load data
    train_data, val_data_short, val_data_long, test_data = load_data()
    start_token_distribution = create_start_token_distribution(train_data)

    # Create Generator
    
    generator = Generator(
        vocab_size=VOCAB_SIZE,
        hidden_dim=config['g_hidden_dim'],
        sequence_length=TRAIN_SEQ_LENGTH,
        start_token_distribution=start_token_distribution,
        device=device,
        num_layers=G_NUM_LAYERS
    )
    
    # Create discriminator
    discriminator = Discriminator(
        vocab_size=VOCAB_SIZE,
        embedding_dim=DISCRIMINATOR_EMB_DIM,
        hidden_dim=DISCRIMINATOR_HIDDEN_DIM,
        dropout_rate=D_DROPOUT_RATE,
        device=device
    )
    
    # Initialize optimizers
    g_optimizer_pretrain = th.optim.Adam(generator.parameters(), lr=config['g_pretrain_lr'])
    d_optimizer_pretrain = th.optim.Adam(discriminator.parameters(), lr=config['d_pretrain_lr'])
    d_optimizer = th.optim.Adam(discriminator.parameters(), lr=config['d_learning_rate'])
    
    gen_weights_path = None
    
    # Pretraining phase
    if config.get('do_pretrain', True):
        
        print("Starting generator pretraining...")

        gen_weights_path = pretrain_generator(
            generator=generator,
            optimizer=g_optimizer_pretrain,
            pre_epoch_num=config['g_pretrain_epochs'],
            batch_size=config['g_pretrain_batch_size'],
            train_data=train_data,
            val_data=val_data_short,  # Use short validation data
            test_data=test_data,
            eval_freq=config['g_eval_pretrain_epochs'],
            lr_patience=G_LR_PATIENCE,
            lr_decay=G_LR_DECAY,
            log_path=gen_pretrain_log
            )

        print("Starting discriminator pretraining...")

        disc_save_path = pretrain_discriminator(
            generator=generator,
            discriminator=discriminator,
            optimizer=d_optimizer_pretrain,
            train_data=train_data, 
            val_data=val_data_short,  # Use short validation data
            batch_size=config['d_batch_size'],
            pretrain_epochs=D_PRETRAIN_EPOCHS,
            log_file=disc_pretrain_log,
            lr_patience=D_LR_PATIENCE,
            lr_decay=D_LR_DECAY,
            min_lr=D_LR_MIN
            )
                
        print(f"Saved pretrained models to {output_dir}")
        
        ## TRANSFER OPTIMIZER STATE ##
        print("Transferring optimizer state from pretraining to adversarial phase...")
        
        pretrain_state = d_optimizer_pretrain.state_dict()
        d_optimizer_state = d_optimizer.state_dict()

        # Copy everything except param_groups (which contains the learning rate)
        for key in pretrain_state.keys():
            if key != 'param_groups':
                d_optimizer_state[key] = pretrain_state[key]
        
        # For param_groups, copy everything except learning rate
        for pg_pretrain, pg_adv in zip(pretrain_state['param_groups'], d_optimizer_state['param_groups']):
            for k in pg_pretrain.keys():
                if k != 'lr':  # Keep all parameters except learning rate
                    pg_adv[k] = pg_pretrain[k]
        
        # Load the modified state
        d_optimizer.load_state_dict(d_optimizer_state)

    else:
        # Load pretrained models if not doing pretraining
        try:
            print("Loading pretrained models...")
            
            # Determine paths based on seed
            seed_prefix = f"{seed}_"
            gen_weights_path = os.path.join(SAVE_DIR, f"{seed_prefix}generator_best_model.pth")
            disc_load_path = os.path.join(SAVE_DIR, f"{seed_prefix}discriminator_best_model.pth")
            
            print(f"Using generator weights from: {gen_weights_path}")
            print(f"Loading discriminator from: {disc_load_path}")
            
            # Load generator
            gen_checkpoint = th.load(gen_weights_path, map_location=device)
            generator.load_state_dict(gen_checkpoint['model_state_dict'])
            
            # Load discriminator
            disc_checkpoint = th.load(disc_load_path, map_location=device)
            discriminator.load_state_dict(disc_checkpoint['model_state_dict'])
            
            # Transfer optimizer state from pretrained discriminator to adversarial discriminator
            print("Transferring optimizer state from pretrained discriminator to adversarial phase...")

            if 'optimizer_state_dict' in disc_checkpoint:
                pretrain_state = disc_checkpoint['optimizer_state_dict']
                d_optimizer_state = d_optimizer.state_dict()
                
                # Copy everything except param_groups (which contains the learning rate)
                for key in pretrain_state.keys():
                    if key != 'param_groups':
                        d_optimizer_state[key] = pretrain_state[key]
                
                # For param_groups, copy everything except learning rate
                for pg_pretrain, pg_adv in zip(pretrain_state['param_groups'], d_optimizer_state['param_groups']):
                    for k in pg_pretrain.keys():
                        if k != 'lr':  # Keep all parameters except learning rate
                            pg_adv[k] = pg_pretrain[k]
                
                # Load the modified state
                d_optimizer.load_state_dict(d_optimizer_state)
                print("Successfully transferred discriminator optimizer state")
            else:
                print("Warning: No optimizer state found in discriminator checkpoint")
            
        except Exception as e:
            print(f"Error loading pretrained models: {e}")
            sys.exit(1)
    
    env = TokenGenerationEnv(
        discriminator=discriminator,
        vocab_size=VOCAB_SIZE,
        seq_length=TRAIN_SEQ_LENGTH,
        start_token_distribution=start_token_distribution,
        device=device
        )

    env = Monitor(env)

    callback = CustomCallback(
        discriminator=discriminator,
        d_optimizer=d_optimizer,
        d_epochs=config.get('d_epochs', 5),
        d_batch_size=config['d_batch_size'],
        train_data=train_data,
        val_data=val_data_short,
        inference_val_data=val_data_long,  # Use val_data_long instead of test_data
        sequence_length=TRAIN_SEQ_LENGTH,
        inference_seq_length=INFERENCE_SEQ_LENGTH,
        start_token_distribution=start_token_distribution,
        vocab_size=VOCAB_SIZE,
        eval_freq=config['eval_freq'],
        verbose=0,
        log_path=ppo_log
        )
    
    # Configure learning rate for PPO
    if config.get('use_linear_lr_decay', False):
        # Use linearly decaying learning rate
        min_lr = config['min_ppo_lr']
        max_lr = config['ppo_learning_rate']
        timesteps = config['ppo_total_timesteps']
        learning_rate = get_linear_fn(min_lr, max_lr, timesteps)
    else:
        # Use constant learning rate
        learning_rate = config['ppo_learning_rate']
    
    # Create PPO model
    ppo_model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=config['ppo_n_steps'],
        batch_size=config['ppo_batch_size'],
        n_epochs=config['ppo_n_epochs'],
        gamma=config['ppo_gamma'],
        gae_lambda=config['ppo_gae_lambda'],
        clip_range=config['ppo_clip_range'],
        clip_range_vf=config['ppo_clip_range_vf'],
        ent_coef=config['ppo_ent_coef'],
        vf_coef=config['ppo_vf_coef'],
        max_grad_norm=config['ppo_max_grad_norm'],
        use_sde=config.get('ppo_use_sde', False),
        verbose=0,
        policy_kwargs=dict(
            lstm_hidden_size=config['g_hidden_dim'],
            n_lstm_layers=G_NUM_LAYERS,
            shared_lstm=True,
            enable_critic_lstm=False,
            net_arch=dict(pi=[], vf=[]),
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=dict(
                betas=(0.95, 0.999)  # Increase beta1 from default 0.9 to 0.95
                )
        )
    )
    
    # If pretrained, transfer weights from generator to PPO model
    if config.get('transfer_weights', True):
        print("Transferring weights from pretrained generator to PPO policy...")
        ppo_model = transfer_weights_from_saved(
            weights_path=gen_weights_path,
            ppo_model=ppo_model,
            transfer_head=config.get('transfer_head', True),
            vocab_size=VOCAB_SIZE,
            hidden_dim=config['g_hidden_dim'],
            sequence_length=TRAIN_SEQ_LENGTH,
            start_token_distribution=start_token_distribution,
            num_layers=G_NUM_LAYERS,
            device=device
        )

    print("Loading best model from training...")
    if config.get('do_hyperparam_search', False):
        config_id = os.getenv('CONFIG_ID', '0')
        best_model_path = os.path.join(MODELS_DIR, f"config_{config_id}_seed_{seed}_best_wasserstein")
        initial_model_path = os.path.join(MODELS_DIR, f"config_{config_id}_seed_{seed}_initial_model")
    else:
        best_model_path = os.path.join(MODELS_DIR, f"{seed}_best_wasserstein")
        initial_model_path = os.path.join(MODELS_DIR, f"{seed}_initial_model")

    # Save initial model right after creation (before training)
    ppo_model.save(initial_model_path)
    print(f"Saved initial model to {initial_model_path}")

    
    # Train with PPO
    print("Starting PPO training...")
    ppo_model.learn(
        total_timesteps=config['ppo_total_timesteps'],
        callback=callback
    )
    
    # Close environment
    env.close()

    print("Loading best model for evaluation...")
    # Check if the best model exists, otherwise use the initial model
    if os.path.exists(best_model_path + ".zip"):
        print(f"Loading best model from {best_model_path}")
        best_model = RecurrentPPO.load(best_model_path, env=None)
    else:
        print(f"No best model found, loading initial model from {initial_model_path}")
        best_model = RecurrentPPO.load(initial_model_path, env=None)

    # Evaluate the best model
    best_model_metrics = evaluate_best_model(
        model=best_model,
        output_path=ppo_log,
        test_data=test_data,
        vocab_size=VOCAB_SIZE,
        inference_seq_length=INFERENCE_SEQ_LENGTH,
        start_token_distribution=start_token_distribution
        )

    # Record training time
    training_time = time.time() - start_time
    
    # Add seed to config for results
    config_with_seed = config.copy()
    config_with_seed['seed'] = seed
    
    # Create results summary
    results = {
        "config": config_with_seed,
        "training_time": training_time,
        'best_model_metrics': best_model_metrics

    }
    
    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed in {training_time:.2f} seconds!")
    

if __name__ == "__main__":
    main()