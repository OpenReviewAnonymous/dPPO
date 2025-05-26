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

# Import local modules
from oracle import Oracle
from generator import Generator, pretrain_generator, transfer_weights_from_saved
from discriminator import Discriminator, evaluate_discriminator, pretrain_discriminator
from environment import TokenGenerationEnv
from callback import CustomCallback

# ============= BASE DIRECTORIES =============
BASE_DIR = Path(os.getenv('WORKING_DIR', Path(os.path.dirname(os.path.abspath(__file__)))))
SAVE_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"
TEXT_DIR = BASE_DIR / "text_file_train"

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# ============= FIXED PARAMETERS =============
# Data parameters
VOCAB_SIZE = 5000
SEQ_LENGTH = 20
START_TOKEN = 0
GENERATED_NUM = 10000  # Number of samples to generate

# Oracle/Generator model parameters
ORACLE_EMB_DIM = 32
ORACLE_HIDDEN_DIM = 32
ORACLE_PARAMS_PATH = SAVE_DIR / 'target_params.pkl'

# GENERATOR
G_NUM_LAYERS = 2
G_LR_PATIENCE = 5
G_LR_DECAY = 0.5

# DISCRIMINATOR
DISCRIMINATOR_EMB_DIM = 64
DISCRIMINATOR_HIDDEN_DIM = 128
D_DROPOUT_RATE = 0.2
D_OUTER_EPOCH = 15
D_INNTER_EPOCH = 3
D_BATCH_SIZE = 128
D_LR_PATIENCE = 10
D_LR_DECAY = 0.5
D_LR_MIN = 1e-5
D_PRETRAIN_LR = 5e-3

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

def main():
    """Main function to run a single PPO-SeqGAN training."""
    
    # Get environment variables
    config_path = os.getenv('CONFIG_PATH')
    seed = int(os.getenv('SEED', '0'))
    output_dir = Path(os.getenv('OUTPUT_DIR', RESULTS_DIR / "ppo_seqgan_runs"))
    
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

    # Create log file paths
    adversarial_log = os.path.join(TEXT_DIR, f"config_{config_id}_seed_{seed}_0_adversarial_training_log.txt")
    gen_pretrain_log = os.path.join(TEXT_DIR, f"config_{config_id}_seed_{seed}_1_generator_pretrain.txt")
    disc_pretrain_log = os.path.join(TEXT_DIR, f"config_{config_id}_seed_{seed}_2_discriminator_pretrain.txt")
    reward_log = os.path.join(TEXT_DIR, f"config_{config_id}_seed_{seed}_3_rewards_log.txt")
    
    # Print training configuration
    print(f"Training PPO-SeqGAN with:")
    print(f"  Seed: {seed}")
    print(f"  Device: {device}")
    print(f"  Generator Hidden Dim: {config['g_hidden_dim']}")
    print(f"  PPO Learning Rate: {config['ppo_learning_rate']}")
    print(f"  Discriminator Learning Rate: {config['d_learning_rate']}")
    print(f"  PPO Total Timesteps: {config['ppo_total_timesteps']}")
    
    # Create Oracle
    oracle = Oracle(
        vocab_size=VOCAB_SIZE,
        embedding_dim=ORACLE_EMB_DIM,
        hidden_dim=ORACLE_HIDDEN_DIM,
        sequence_length=SEQ_LENGTH,
        start_token=START_TOKEN,
        device=device
    )
    
    # Load oracle parameters
    print(f"Loading oracle parameters from {ORACLE_PARAMS_PATH}...")
    try:
        oracle.load_params(ORACLE_PARAMS_PATH)
        if not os.path.exists(ORACLE_PARAMS_PATH):
            raise FileNotFoundError(f"Oracle parameter file not found: {ORACLE_PARAMS_PATH}")
    except Exception as e:
        print(f"Error loading oracle parameters: {e}")
        sys.exit(1)  # Exit with error code
    
    # Generate positive samples from oracle once
    print("Generating real data from oracle (target LSTM)...")
    oracle.eval()
    with th.no_grad():
        positive_samples = oracle.generate(GENERATED_NUM)
    print(f"Generated {GENERATED_NUM} positive samples.")
    
    # Create Generator for pretraining
    generator = Generator(
        vocab_size=VOCAB_SIZE,
        hidden_dim=config['g_hidden_dim'],
        sequence_length=SEQ_LENGTH,
        start_token=START_TOKEN,
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
    d_pretrain_optimizer = th.optim.Adam(discriminator.parameters(), lr=D_PRETRAIN_LR)

    d_optimizer = th.optim.Adam(discriminator.parameters(), lr=config['d_learning_rate'])

    gen_weights_path = None
    
    # Pretraining phase
    if config.get('do_pretrain', True):
        print("Starting generator pretraining...")
        
        pretrain_generator(
            target_lstm=oracle,
            generator=generator,
            optimizer=g_optimizer_pretrain,
            pre_epoch_num=config['g_pretrain_epochs'],
            batch_size=config['g_pretrain_batch_size'],
            generated_num=GENERATED_NUM,
            positive_samples=positive_samples,
            eval_freq=config['g_eval_pretrain_epochs'],
            lr_patience=G_LR_PATIENCE,
            lr_decay=G_LR_DECAY,
            log_path=gen_pretrain_log
        )

        print("Starting discriminator pretraining...")
        
        pretrain_discriminator(
            target_lstm=oracle,
            generator=generator,
            discriminator=discriminator,
            optimizer=d_pretrain_optimizer,
            outer_epochs=D_OUTER_EPOCH,
            inner_epochs=D_INNTER_EPOCH,
            batch_size=D_BATCH_SIZE,
            generated_num=GENERATED_NUM,
            positive_samples=positive_samples,
            log_file=disc_pretrain_log,
            lr_patience=D_LR_PATIENCE,
            lr_decay=D_LR_DECAY,
            min_lr=D_LR_MIN
        )

        # Save pretrained models
        gen_weights_path = os.path.join(output_dir, f"{seed}_generator_pretrained.pth")
        disc_save_path = os.path.join(output_dir, f"{seed}_discriminator_pretrained.pth")
        
        # Save generator
        th.save({
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': g_optimizer_pretrain.state_dict()
        }, gen_weights_path)
        
        # Save discriminator
        th.save({
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': d_pretrain_optimizer.state_dict()
        }, disc_save_path)
        
        
        pretrain_state = d_pretrain_optimizer.state_dict()
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
            gen_weights_path = os.path.join(SAVE_DIR, f"{seed_prefix}generator_pretrained.pth")
            disc_load_path = os.path.join(SAVE_DIR, f"{seed_prefix}discriminator_pretrained.pth")
            
            print(f"Using generator weights from: {gen_weights_path}")
            print(f"Loading discriminator from: {disc_load_path}")
            
            # Load generator (only needed for the discriminator training)
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

    
    # Set up the environment for PPO
    env = TokenGenerationEnv(
        discriminator=discriminator,
        vocab_size=VOCAB_SIZE,
        seq_length=SEQ_LENGTH,
        start_token=START_TOKEN,
        device=device
    )
    env = Monitor(env)

    # Set up callback for discriminator training during PPO
    callback = CustomCallback(
        discriminator=discriminator,
        oracle=oracle,
        d_optimizer=d_optimizer,
        d_steps=config['d_steps'],
        k_epochs=config['k_epochs'],
        d_batch_size=D_BATCH_SIZE,
        positive_samples=positive_samples,
        sequence_length=SEQ_LENGTH,
        start_token=START_TOKEN,
        generated_num=GENERATED_NUM,
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
            shared_lstm=False,
            enable_critic_lstm=True,
            net_arch=dict(pi=[], vf=[]),
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=dict(
                betas=(0.95, 0.999)
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
            sequence_length=SEQ_LENGTH,
            start_token=START_TOKEN,
            num_layers=G_NUM_LAYERS,
            device=device
        )

    
    # Train with PPO
    print("Starting PPO training...")
    ppo_model.learn(
        total_timesteps=config['ppo_total_timesteps'],
        callback=callback
    )
    
    # Record training time
    training_time = time.time() - start_time
    
    # Add seed to config for results
    config_with_seed = config.copy()
    config_with_seed['seed'] = seed
    
    # Create results summary
    results = {
        "config": config_with_seed,
        "training_time": training_time
    }
    
    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed in {training_time:.2f} seconds!")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()