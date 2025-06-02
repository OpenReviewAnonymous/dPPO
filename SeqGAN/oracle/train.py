import os
import time
import json
import random
import numpy as np
import torch as th
from pathlib import Path
import sys
from torch.utils.data import DataLoader, TensorDataset

# Import models
from oracle import Oracle
from generator import Generator, generator_adversarial_update, pretrain_generator
from discriminator import Discriminator, evaluate_discriminator, pretrain_discriminator
from rollout import Rollout

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
GENERATED_NUM = 10000

# Oracle/Generator model parameters
ORACLE_EMB_DIM = 32
ORACLE_HIDDEN_DIM = 32
ORACLE_PARAMS_PATH = SAVE_DIR / 'target_params.pkl'

# ROLLOUT
ROLLOUT_NUM = 16
ROLLOUT_UPDATE_RATE = 0.8

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

def train_seqgan(generator, discriminator, rollout, target_lstm, g_optimizer, d_optimizer, 
                num_epochs, batch_size, generated_num, positive_samples, g_steps, d_steps, k_epochs, 
                log_path, log_path_reward, device):
    
    # Open log file
    log = open(log_path, 'w')
    log.write('epoch\tnll\tpg_loss\td_loss\td_accuracy\treal_prob\tfake_prob\tavg_reward\n')
    
    # Optional: Create a separate reward log file if needed
    reward_log = open(log_path_reward, 'w')
    reward_log.write('epoch\tposition\treward\n')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        
        # GENERATOR TRAINING PHASE
        pg_losses = []
        avg_token_rewards = []
        
        for _ in range(g_steps):
            # Generate sequences
            generator.train()
            sequences = generator.generate(batch_size)
            
            # Get rewards using rollout
            rewards = rollout.get_reward(sequences)
            
            # Track per-token rewards
            avg_position_rewards = rewards.mean(dim=0).cpu().numpy()    # Average across batch for each position
            avg_token_rewards.append(avg_position_rewards.mean())       # Overall average reward

            # Log per-position rewards for EVERY epoch
            for pos, reward in enumerate(avg_position_rewards):
                reward_log.write(f"{epoch}\t{pos}\t{reward:.6f}\n")
            
            # Update generator using policy gradient
            pg_loss = generator_adversarial_update(generator, sequences, rewards, g_optimizer)
            pg_losses.append(pg_loss)
        
        # Flush the reward log every epoch to ensure data is written
        reward_log.flush()
        
        # Average PG loss for this epoch
        avg_pg_loss = sum(pg_losses) / len(pg_losses) if pg_losses else 0
        avg_reward = sum(avg_token_rewards) / len(avg_token_rewards) if avg_token_rewards else 0
        
        # Update rollout module with new generator parameters
        rollout.update_params()
        
        # DISCRIMINATOR TRAINING PHASE
        d_losses = []
        
        for _ in range(d_steps):
            # Generate new data for discriminator training
            generator.eval()
            negative_examples = generator.generate(generated_num)
            
            # Create data loaders
            pos_loader = DataLoader(TensorDataset(positive_samples), batch_size=batch_size, shuffle=True)
            neg_loader = DataLoader(TensorDataset(negative_examples), batch_size=batch_size, shuffle=True)
            
            # Train discriminator for k epochs
            for _ in range(k_epochs):
                
                discriminator.train()
                
                batch_d_losses = []
                for (pos_batch,), (neg_batch,) in zip(pos_loader, neg_loader):
                    d_loss = discriminator.train_step(pos_batch, neg_batch, d_optimizer)
                    batch_d_losses.append(d_loss)
                
                if batch_d_losses:
                    d_losses.append(sum(batch_d_losses) / len(batch_d_losses))
        
        # Average discriminator loss
        avg_d_loss = sum(d_losses) / len(d_losses) if d_losses else 0
        
        # EVALUATION PHASE
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            
            generator.eval()
            
            # Generate samples for evaluation
            eval_samples = generator.generate(generated_num//10)
            # Calculate NLL using oracle
            nll = target_lstm.calculate_nll(eval_samples)
            
            # Evaluate discriminator performance
            disc_metrics = evaluate_discriminator(discriminator, target_lstm, generator, num_samples=generated_num//10)
            d_accuracy = disc_metrics['accuracy']
            real_prob = disc_metrics['real_prob']
            fake_prob = disc_metrics['fake_prob']
            
            # Log all metrics
            metrics_str = f"Epoch {epoch}, NLL: {nll:.4f}, PG Loss: {avg_pg_loss:.4f}, D Loss: {avg_d_loss:.4f}, "
            metrics_str += f"D Acc: {d_accuracy:.4f}, Real Prob: {real_prob:.4f}, Fake Prob: {fake_prob:.4f}, Avg Reward: {avg_reward:.4f}"
            print(metrics_str)
            
            # Write to log file
            log.write(f"{epoch}\t{nll:.4f}\t{avg_pg_loss:.4f}\t{avg_d_loss:.4f}\t{d_accuracy:.4f}\t{real_prob:.6f}\t{fake_prob:.4f}\t{avg_reward:.4f}\n")
            log.flush()
    
    log.close()
    reward_log.close()
    print("Training completed!")

def main():
    """Main function to run a single generator training."""
    
    # Get environment variables
    config_path = os.getenv('CONFIG_PATH')
    seed = int(os.getenv('SEED', '0'))
    config_id = int(os.getenv('CONFIG_ID', '-1'))
    output_dir = Path(os.getenv('OUTPUT_DIR', RESULTS_DIR))
    
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
    print(f"Training generator with:")
    print(f"  Seed: {seed}")
    print(f"  Device: {device}")
    print(f"  Batch Size: {config['g_adv_batch_size']}")
    print(f"  Generator Learning Rate: {config['g_learning_rate']}")
    print(f"  Discriminator Learning Rate: {config['d_learning_rate']}")
    print(f"  Pre-training Epochs: {config['pretrain_epochs']}")
    print(f"  Total Adversarial Epochs: {config['adv_epochs']}")
    
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

    # Generate positive samples
    print("Generating real data from oracle (target LSTM)...")
    oracle.eval()
    with th.no_grad():
        positive_samples = oracle.generate(GENERATED_NUM)
    print(f"Generated {GENERATED_NUM} positive samples.")
    
    
    # Create Generator
    generator = Generator(
        vocab_size=VOCAB_SIZE,
        hidden_dim=config['g_hidden_dim'],
        sequence_length=SEQ_LENGTH,
        start_token=START_TOKEN,
        num_layers=config['g_num_layers'],
        device=device
    )
    
    # Create discriminator
    discriminator = Discriminator(
        vocab_size=VOCAB_SIZE,
        embedding_dim=config['d_emb_dim'],
        hidden_dim=config['d_hidden_dim'],
        dropout_rate=config['d_dropout'],
        num_layers=config['d_num_layers'],
        device=device
    )
    
    # Initialize optimizers
    g_optimizer_pretrain = th.optim.Adam(generator.parameters(), lr=config['g_lr_pretrain'])
    d_optimizer_pretrain = th.optim.Adam(discriminator.parameters(), lr=config['d_lr_pretrain'])

    # Adversarial training optimizers
    g_optimizer = th.optim.Adam(generator.parameters(), lr=config['g_learning_rate'])
    d_optimizer = th.optim.Adam(discriminator.parameters(), lr=config['d_learning_rate'])

    gen_weights_path = None

    # Pretraining phase
    if config.get('do_pretrain', True):

        print("Starting generator pretraining...")

        pretrain_generator(
            target_lstm=oracle,
            generator=generator,
            optimizer=g_optimizer_pretrain,
            pre_epoch_num=config['pretrain_epochs'],
            batch_size=config['g_pretrain_batch_size'],
            generated_num=GENERATED_NUM,
            positive_samples=positive_samples,
            eval_freq=config['g_pre_eval_freq'],
            lr_patience=config['g_lr_patience'],
            lr_decay=config['g_lr_decay'],
            log_path=gen_pretrain_log
        )

        print("Starting discriminator pretraining...")
    
        pretrain_discriminator(
            target_lstm=oracle,
            generator=generator,
            discriminator=discriminator,
            optimizer=d_optimizer_pretrain,
            outer_epochs=config['d_outer_epochs'],
            inner_epochs=config['d_inner_epochs'],
            batch_size=config['d_batch_size'],
            generated_num=GENERATED_NUM,
            positive_samples=positive_samples,
            log_file=disc_pretrain_log,
            lr_patience=config['d_lr_patience'],
            lr_decay=config['d_lr_decay'],
            min_lr=config['d_lr_min'],
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
            'optimizer_state_dict': d_optimizer_pretrain.state_dict()
        }, disc_save_path)
        
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
    
    
    # Initialize rollout module
    rollout = Rollout(
        generator=generator,
        discriminator=discriminator,
        rollout_num=ROLLOUT_NUM,
        update_rate=ROLLOUT_UPDATE_RATE,
        device=device
    )

    # Start adversarial training
    print("Starting adversarial training...")
    
    train_seqgan(
        generator=generator,
        discriminator=discriminator,
        rollout=rollout,
        target_lstm=oracle,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        num_epochs=config['adv_epochs'],
        batch_size=config['g_adv_batch_size'],
        generated_num=GENERATED_NUM,
        positive_samples=positive_samples,
        g_steps=config['g_steps'],
        d_steps=config['d_steps'],
        k_epochs=config['k_epochs'],
        log_path=adversarial_log,
        log_path_reward=reward_log,
        device=device
    )
        
    # Record training time
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds!")

if __name__ == "__main__":
    main()