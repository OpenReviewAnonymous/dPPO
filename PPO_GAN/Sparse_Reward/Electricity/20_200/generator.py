import os
from pathlib import Path
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scipy.stats import wasserstein_distance
from scipy.special import kl_div

class Generator(nn.Module):
    
    def __init__(self, vocab_size, hidden_dim, sequence_length, start_token_distribution, device, num_layers=2):
        
        super(Generator, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token_distribution = start_token_distribution
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)    # Action head

        # Initialize on device
        self.to(self.device)
    
    def _to_onehot(self, tokens):
        """Convert batch of tokens to one-hot vectors"""
        batch_size, seq_length = tokens.shape
        onehot = th.zeros(batch_size, seq_length, self.vocab_size, device=self.device)
        return onehot.scatter_(2, tokens.unsqueeze(-1), 1)
    
    def _to_onehot_single(self, token):
        """Convert single token to one-hot vector for generation"""
        onehot = th.zeros(1, 1, self.vocab_size, device=self.device)
        return onehot.scatter_(2, token.view(1, 1, 1), 1)
       
    def forward(self, x, hidden=None):

        if x.dim() == 2:                            # [batch_size, sequence_length]
            x_onehot = self._to_onehot(x)           # [batch_size, sequence_length, vocab_size]
        else:  
            x_onehot = self._to_onehot_single(x)    # Single token [batch_size, 1]
        
        lstm_out, hidden = self.lstm(x_onehot, hidden)  # lstm_out: [batch_size, sequence_length, hidden_dim]
        logits = self.fc(lstm_out)             # Output layer
        
        return logits, hidden
    
    def generate(self, num_samples, sequence_length=None):
        
        with th.no_grad():
            
            seq_length = sequence_length if sequence_length is not None else self.sequence_length
        
            start_tokens = np.random.choice(
                np.arange(self.vocab_size), 
                size=num_samples, 
                p=self.start_token_distribution
                )

            # Start token for all sequences
            x = th.tensor(start_tokens, dtype=th.long, device=self.device).unsqueeze(1)
            hidden = None  # Let PyTorch initialize the hidden state

            generated_sequences = th.zeros(num_samples, seq_length, dtype=th.long, device=self.device)

            for i in range(seq_length):
                # Forward pass
                logits, hidden = self.forward(x[:, -1:], hidden)
                
                # Sample from distribution
                probs = F.softmax(logits.squeeze(1), dim=-1)
                next_token = th.multinomial(probs, 1)
                
                # Add to sequence
                generated_sequences[:, i] = next_token.squeeze()
                
                # Update input for next step
                x = next_token
            
            return generated_sequences

    def pretrain_step(self, x, optimizer):
        
        optimizer.zero_grad()
            
        inputs = x[:, :-1]                  # Input is all tokens except last one
        targets = x[:, 1:].contiguous()     # Target is all tokens except first one (shifted by 1)
        
        logits, _ = self.forward(inputs)
    
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        return loss.item()


def pretrain_generator(generator, optimizer, pre_epoch_num, batch_size, 
                      train_data, val_data, test_data, eval_freq, 
                      lr_patience, lr_decay, log_path):
    
    print('Start pre-training...')

    # Open log file
    log = open(log_path, 'w')
    log.write('pre-training...\n')

    # For learning rate scheduling
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Create DataLoaders
    train_dataset = th.utils.data.TensorDataset(th.tensor(train_data, dtype=th.long))
    val_dataset = th.utils.data.TensorDataset(th.tensor(val_data, dtype=th.long))
    
    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = th.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    for epoch in range(pre_epoch_num):
        # Training phase
        generator.train()
        epoch_train_loss = 0
        train_batch_count = 0

        for batch_data in train_loader:
            x = batch_data[0].to(generator.device)
            loss = generator.pretrain_step(x, optimizer)
            epoch_train_loss += loss
            train_batch_count += 1
            
        avg_train_loss = epoch_train_loss / train_batch_count if train_batch_count > 0 else float('inf')
        
        # Validation phase
        generator.eval()
        epoch_val_loss = 0
        val_batch_count = 0
        
        with th.no_grad():
            for batch_data in val_loader:
                x = batch_data[0].to(generator.device)
                # Use inputs and targets as in pretrain_step but without optimizer updates
                inputs = x[:, :-1]
                targets = x[:, 1:].contiguous()
                logits, _ = generator.forward(inputs)
                loss = th.nn.functional.cross_entropy(logits.reshape(-1, generator.vocab_size), 
                                                      targets.reshape(-1))
                epoch_val_loss += loss.item()
                val_batch_count += 1
                
        avg_val_loss = epoch_val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        
        # Evaluate frequently and log results
        if epoch % eval_freq == 0 or epoch == pre_epoch_num - 1:
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            buffer = f'epoch:\t{epoch}\ttrain_loss:\t{avg_train_loss:.5f}\tval_loss:\t{avg_val_loss:.5f}\n'
            log.write(buffer)
            log.flush()
        
        # Learning rate scheduling and model saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss
            }
            print(f"New best model at epoch {epoch} with validation loss: {avg_val_loss:.5f}")
        else:
            patience_counter += 1
            
        if patience_counter >= lr_patience:
            # Reduce learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            print(f"Learning rate reduced to {optimizer.param_groups[0]['lr']}")
            patience_counter = 0

    # Save the best model
    seed_str = Path(log_path).stem.split('_')[0]  # Extract seed from log filename
    models_dir = Path(os.getenv('MODELS_DIR', './saved_models_training'))
    best_model_path = str(models_dir / f"{seed_str}_generator_best_model.pth")

    #best_model_path = log_path.replace('.txt', '_best_model.pth')
    th.save(best_model_state, best_model_path)
    print(f"Best model saved to {best_model_path}")
    
    # Load the best model for final evaluation
    generator.load_state_dict(best_model_state['model_state_dict'])
    
    # Final evaluation with distribution metrics
    generator.eval()
    test_tensor = th.tensor(test_data, dtype=th.long).to(generator.device)
    
    # Generate sequences matching test data size
    with th.no_grad():
        inference_seq_length = test_data.shape[1]
        generated_sequences = generator.generate(len(test_data), sequence_length=inference_seq_length)
    
    # Convert to numpy for distance calculation
    generated_np = generated_sequences.cpu().numpy()
    test_np = test_tensor.cpu().numpy()
    
    # Calculate Wasserstein distances and KL divergence for each timestep
    seq_length = test_np.shape[1]
    wasserstein_distances = []
    kl_divergences = []
    
    for t in range(seq_length):
        # Calculate standard deviation of real data at this timestep
        real_std = np.std(test_np[:, t])
        
        # Calculate Wasserstein distance and normalize by standard deviation
        w_dist = wasserstein_distance(test_np[:, t], generated_np[:, t])
        normalized_w_dist = w_dist / real_std if real_std > 0 else float('inf')
        wasserstein_distances.append(normalized_w_dist)
        
        # Calculate KL divergence
        # First get probability distributions by using histograms
        vocab_size = generator.vocab_size
        bins = np.arange(0, vocab_size + 1) - 0.5  # Create bins for each token value
        
        real_hist, _ = np.histogram(test_np[:, t], bins=bins, density=True)
        gen_hist, _ = np.histogram(generated_np[:, t], bins=bins, density=True)
        
        # Smooth probabilities to avoid division by zero
        epsilon = 1e-10
        real_hist = real_hist + epsilon
        gen_hist = gen_hist + epsilon
        
        # Normalize to ensure they sum to 1
        real_hist = real_hist / real_hist.sum()
        gen_hist = gen_hist / gen_hist.sum()
        
        # Calculate KL divergence
        kl = np.sum(kl_div(real_hist, gen_hist))
        kl_divergences.append(kl)
    
    # Save metrics to file
    metrics_path = str(Path(log_path).parent / f"{seed_str}_inference_pretrain.txt")
    #metrics_path = log_path.replace('.txt', '_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("timestep\tnormalized_wasserstein\tkl_divergence\n")
        for t in range(seq_length):
            f.write(f"{t}\t{wasserstein_distances[t]:.6f}\t{kl_divergences[t]:.6f}\n")
    
    # Log average metrics
    avg_wasserstein = np.mean(wasserstein_distances)
    avg_kl = np.mean(kl_divergences)
    log.write(f"Final average normalized Wasserstein distance: {avg_wasserstein:.6f}\n")
    log.write(f"Final average KL divergence: {avg_kl:.6f}\n")

    ## Save generated samples
    samples_path = str(Path(log_path).parent / f"{seed_str}_inference_pretrain.npy")
    np.save(samples_path, generated_np)
    print(f"Generated samples from pretraining saved to {samples_path}")
    
    log.close()
    
    print(f'Pretraining finished! Final metrics: Wasserstein={avg_wasserstein:.6f}, KL={avg_kl:.6f}')
    return best_model_path

def transfer_weights_from_saved(weights_path, ppo_model, transfer_head, vocab_size, hidden_dim, sequence_length, start_token_distribution, num_layers, device):

    # Create temporary supervised model to load weights into
    temp_generator = Generator(vocab_size, hidden_dim, sequence_length, start_token_distribution, device, num_layers)
    
    # Load the saved weights
    saved_weights = th.load(weights_path, weights_only=False)
    temp_generator.load_state_dict(saved_weights['model_state_dict'])    
    
    # Transfer LSTM weights
    print("\n=== Transferring LSTM Weights ===")
    supervised_state_dict = temp_generator.state_dict()
    ppo_lstm_dict = ppo_model.policy.lstm_actor.state_dict()
    
    # Print shapes before transfer for verification
    print("\nWeight shapes before transfer:")
    print("\nSupervised LSTM weights:")
    for key, value in supervised_state_dict.items():
        if 'lstm' in key:
            print(f"{key}: {value.shape}")
    
    print("\nPPO LSTM weights:")
    for key, value in ppo_lstm_dict.items():
        print(f"{key}: {value.shape}")
    
    # Transfer LSTM weights
    lstm_transfer_count = 0
    for ppo_key in ppo_lstm_dict.keys():
        supervised_key = f"lstm.{ppo_key}"
        if supervised_key in supervised_state_dict:
            if ppo_lstm_dict[ppo_key].shape == supervised_state_dict[supervised_key].shape:
                ppo_lstm_dict[ppo_key].copy_(supervised_state_dict[supervised_key])
                lstm_transfer_count += 1
                print(f"Transferred weights for {ppo_key}")
            else:
                print(f"Shape mismatch for {ppo_key}")
    
    # Load the LSTM weights
    ppo_model.policy.lstm_actor.load_state_dict(ppo_lstm_dict)
    print(f"\nSuccessfully transferred {lstm_transfer_count} LSTM weight tensors")
    
    # Transfer head weights if requested
    if transfer_head:
        print("\n=== Transferring Head Weights ===")
        # Get supervised fc weights and biases
        fc_weight = supervised_state_dict['fc.weight']
        fc_bias = supervised_state_dict['fc.bias']
        
        # Get PPO action_net weights and biases
        action_net_state_dict = ppo_model.policy.action_net.state_dict()
        
        print("\nHead weight shapes:")
        print(f"Supervised fc weight: {fc_weight.shape}")
        print(f"Supervised fc bias: {fc_bias.shape}")
        print(f"PPO action_net weight: {action_net_state_dict['weight'].shape}")
        print(f"PPO action_net bias: {action_net_state_dict['bias'].shape}")
        
        # Verify shapes match before transfer
        if (fc_weight.shape == action_net_state_dict['weight'].shape and 
            fc_bias.shape == action_net_state_dict['bias'].shape):
            # Transfer weights
            action_net_state_dict['weight'].copy_(fc_weight)
            action_net_state_dict['bias'].copy_(fc_bias)
            ppo_model.policy.action_net.load_state_dict(action_net_state_dict)
            print("Successfully transferred head weights")
        else:
            print("Shape mismatch in head weights - transfer aborted")
    
    return ppo_model


def generate_ppo_samples(model, start_token_distribution, sequence_length, num_samples):
    """Generate sequence samples using the model."""
    model.policy.set_training_mode(False)

    start_tokens = np.random.choice(
        np.arange(len(start_token_distribution)), 
        size=num_samples, 
        p=start_token_distribution
        )

    obs = start_tokens
    lstm_states = None
    episode_starts = np.ones((num_samples,), dtype=bool)
    
    # Initialize sequences
    sequences = [[] for _ in range(num_samples)]

    # Generate full sequence_length tokens
    for _ in range(sequence_length):
        actions, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
        for i, action in enumerate(actions):
            sequences[i].append(int(action))
        obs = actions
        episode_starts = np.zeros((num_samples,), dtype=bool)
    
    # Convert to numpy array
    return np.array(sequences)

def calculate_ppo_metrics(real_data, generated_data, stds, vocab_size):
    """Calculate raw and normalized Wasserstein distances and KL divergence."""
    
    raw_wasserstein_distances = []
    norm_wasserstein_distances = []
    kl_divergences = []
    per_timestep_metrics = []  # Store all metrics for each timestep
    
    sequence_length = real_data.shape[1]
    
    # Calculate metrics for each timestep
    for t in range(sequence_length):
        # Get data for this timestep
        real_t = real_data[:, t]
        gen_t = generated_data[:, t]
        
        # Calculate raw Wasserstein distance
        w_dist = wasserstein_distance(real_t, gen_t)
        raw_wasserstein_distances.append(w_dist)
        
        # Calculate normalized Wasserstein distance
        real_std = stds[t]
        norm_w_dist = w_dist / real_std if real_std > 0 else float('inf')
        norm_wasserstein_distances.append(norm_w_dist)
        
        # Calculate KL divergence
        bins = np.arange(0, vocab_size + 1) - 0.5  # Create bins for each token value
        
        real_hist, _ = np.histogram(real_t, bins=bins, density=True)
        gen_hist, _ = np.histogram(gen_t, bins=bins, density=True)
        
        # Smooth probabilities to avoid division by zero
        epsilon = 1e-10
        real_hist = real_hist + epsilon
        gen_hist = gen_hist + epsilon
        
        # Normalize to ensure they sum to 1
        real_hist = real_hist / real_hist.sum()
        gen_hist = gen_hist / gen_hist.sum()
        
        # Calculate KL divergence
        kl = np.sum(kl_div(real_hist, gen_hist))
        kl_divergences.append(kl)
        
        # Store all metrics for this timestep
        per_timestep_metrics.append((w_dist, norm_w_dist, kl))
    
    # Return average metrics and per-timestep metrics
    return {
        'wasserstein_raw': float(np.mean(raw_wasserstein_distances)),
        'wasserstein_norm': float(np.mean(norm_wasserstein_distances)),
        'kl_divergence': float(np.mean(kl_divergences))
    }, per_timestep_metrics

def evaluate_best_model(model, output_path, test_data, vocab_size, inference_seq_length, start_token_distribution):
    """Evaluate the provided model on test data."""
    
    print(f"Evaluating model... (inference sequence length: {inference_seq_length})")
    
    # Convert test data to tensor and calculate standard deviations
    test_tensor = th.tensor(test_data, dtype=th.float32)
    test_stds = th.std(test_tensor, dim=0).cpu().numpy()
    
    # Generate samples using the model
    test_samples = generate_ppo_samples(model, start_token_distribution, inference_seq_length, len(test_data))
    
    # Calculate metrics
    metrics, per_timestep_metrics = calculate_ppo_metrics(test_data, test_samples, test_stds, vocab_size)
    
    # Extract seed from output_path
    if "config_" in str(output_path):
        # Format: config_{config_id}_seed_{seed}_adversarial_train.txt
        parts = Path(output_path).stem.split('_')
        config_id = parts[1]
        seed_str = parts[3]
        prefix = f"config_{config_id}_seed_{seed_str}_"
    else:
        # Format: {seed}_adversarial_train.txt
        seed_str = Path(output_path).stem.split('_')[0]
        prefix = f"{seed_str}_"
    
    # Save the generated samples to the models directory
    models_dir = Path(os.getenv('MODELS_DIR', './saved_models_training'))
    samples_path = str(models_dir / f"{prefix}inference_adversial.npy")
    np.save(samples_path, test_samples)
    print(f"Generated samples saved to {samples_path}")
    
    # Write per-timestep metrics to metrics directory
    metrics_dir = Path(os.getenv('METRICS_DIR', './saved_metrics_training'))
    metrics_path = str(metrics_dir / f"{prefix}inference_adversial.txt")
    
    # Write all metrics including KL divergence
    with open(metrics_path, 'w') as f:
        f.write("timestep\traw_wasserstein\tnormalized_wasserstein\tkl_divergence\n")
        for t, (raw, norm, kl) in enumerate(per_timestep_metrics):
            f.write(f"{t}\t{raw:.6f}\t{norm:.6f}\t{kl:.6f}\n")
    
    # Print summary metrics
    print(f"Model metrics: Raw Wasserstein={metrics['wasserstein_raw']:.6f}, " 
          f"Normalized={metrics['wasserstein_norm']:.6f}, "
          f"KL Divergence={metrics['kl_divergence']:.6f}")
    print(f"Per-timestep metrics saved to {metrics_path}")
    
    return metrics

