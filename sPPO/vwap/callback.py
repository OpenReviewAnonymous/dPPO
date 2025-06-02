import os
import numpy as np
import torch as th
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.callbacks import BaseCallback
from scipy.stats import wasserstein_distance
from scipy.special import kl_div

class CustomCallback(BaseCallback):
    
    def __init__(self, discriminator, d_optimizer, d_epochs, d_batch_size,
                 train_data, val_data, inference_val_data, sequence_length, inference_seq_length,
                 start_token_distribution, vocab_size, eval_freq=1, verbose=0, log_path=None):
        
        super(CustomCallback, self).__init__(verbose)
        
        # Models and optimizer
        self.discriminator = discriminator
        self.d_optimizer = d_optimizer
        
        # Training parameters
        self.d_epochs = d_epochs
        self.d_batch_size = d_batch_size
        self.eval_freq = eval_freq
        self.vocab_size = vocab_size
        
        # Data - keep only train and validation data
        self.train_data = th.tensor(train_data, dtype=th.long, device=discriminator.device)
        self.val_data = th.tensor(val_data, dtype=th.long, device=discriminator.device)
        self.num_train_samples = len(self.train_data)
        self.start_token_distribution = start_token_distribution
        self.sequence_length = sequence_length

        self.inference_val_data = th.tensor(inference_val_data, dtype=th.long, device=discriminator.device)
        self.inference_seq_length = inference_seq_length
        
        # Calculate standard deviations for validation data only
        self.val_stds = th.std(self.val_data.float(), dim=0).cpu().numpy()
        self.inference_val_stds = th.std(self.inference_val_data.float(), dim=0).cpu().numpy()
        
        # Initialize tracking metrics
        self.rollout_count = 0
        self.best_wasserstein = float('inf')
        self.initial_kl = float('inf')
        self.best_kl = float('inf')
        
        # Log path
        self.log_path = log_path if log_path else '0_ppo_adversarial_training.txt'

        ### NEW
        # Check if path contains "config_" pattern and extract seed accordingly
        if "config_" in str(self.log_path):
            # Format: config_{config_id}_seed_{seed}_adversarial_train.txt
            parts = Path(self.log_path).stem.split('_')
            config_id = parts[1]
            seed_str = parts[3]
            self.model_path = Path(os.getenv('MODELS_DIR', './saved_models_training')) / f"config_{config_id}_seed_{seed_str}_best_wasserstein"
        else:
            # Format: {seed}_adversarial_train.txt
            seed_str = Path(self.log_path).stem.split('_')[0]
            self.model_path = Path(os.getenv('MODELS_DIR', './saved_models_training')) / f"{seed_str}_best_wasserstein"
        ### NEW

        with open(self.log_path, 'w') as f:
            f.write('rollout\tlong_wass_norm\tlong_kl\tshort_wass_norm\tshort_kl\tpolicy_loss\tvalue_loss\tentropy\td_loss\td_accuracy\treal_prob\tfake_prob\tavg_reward\n')
        
    def _on_training_start(self):
        """Called at the start of training"""
        pass
        
    def _on_rollout_start(self):
        """Called at the start of a rollout"""
        pass

    def _on_rollout_end(self):
        """Called at the end of a rollout - this is where we'll train the discriminator"""

        self.rollout_count += 1

        # Train the discriminator on a limited number of batches (using short sequences)
        d_loss = self._train_discriminator()

        # Only evaluate and log after eval_freq rollouts
        if self.rollout_count % self.eval_freq == 0:
            # === SHORT SEQUENCE EVALUATION ===
            # Generate validation-sized sample set for evaluation on short sequences
            short_val_samples = self._generate_samples(len(self.val_data))
            
            # Calculate metrics on validation data (short sequences)
            short_wasserstein_raw, short_wasserstein_norm, short_kl = self._calculate_metrics(
                self.val_data.cpu().numpy(), 
                short_val_samples.cpu().numpy(),
                self.sequence_length,
                self.val_stds
                )
            
            # === LONG SEQUENCE EVALUATION ===
            # Generate validation-sized sample set for long sequence evaluation
            long_val_samples = self._generate_samples_with_length(len(self.inference_val_data), self.inference_seq_length)
            
            # Calculate metrics on long validation data
            long_wasserstein_raw, long_wasserstein_norm, long_kl = self._calculate_metrics(
                self.inference_val_data.cpu().numpy(), 
                long_val_samples.cpu().numpy(),
                self.inference_seq_length,
                self.inference_val_stds
                )
            
            # Get discriminator evaluation metrics (on short sequences)
            disc_metrics = self._evaluate_discriminator(short_val_samples)

            # Get current PPO metrics directly from logger
            policy_loss = self.logger.name_to_value.get('train/policy_gradient_loss', 0)
            value_loss = self.logger.name_to_value.get('train/value_loss', 0)
            entropy = self.logger.name_to_value.get('train/entropy_loss', 0)

            # Calculate average reward per episode from rollout buffer
            avg_reward = self._calculate_average_reward()

            # Log to file after each evaluation (including both short and long metrics)
            with open(self.log_path, 'a') as f:
                f.write(f'{self.rollout_count}\t{long_wasserstein_norm:.6f}\t{long_kl:.6f}\t'
                    f'{short_wasserstein_norm:.6f}\t{short_kl:.6f}\t'
                    f'{policy_loss:.6f}\t{value_loss:.6f}\t{entropy:.6f}\t'
                    f'{d_loss:.6f}\t{disc_metrics["accuracy"]:.6f}\t{disc_metrics["real_prob"]:.6f}\t{disc_metrics["fake_prob"]:.6f}\t{avg_reward:.6f}\n')
                f.flush()
            
            if self.rollout_count == 1:
                # Set initial KL value after first evaluation (for long sequences)
                self.initial_kl = long_kl
                print(f"Initial KL divergence (long sequences): {self.initial_kl:.6f}")
            
            # Save model if LONG sequence Wasserstein improves AND KL is below threshold
            if long_wasserstein_norm < self.best_wasserstein and long_kl < self.initial_kl and self.rollout_count > 1:
                self.best_wasserstein = long_wasserstein_norm
                self.best_kl = long_kl  # Track best KL that meets criteria
                
                self.model.save(str(self.model_path))
                print(f"New best model saved with Long Wasserstein: {long_wasserstein_norm:.6f}, Long KL: {long_kl:.6f}")

    def _on_training_end(self):
        """Called at the end of training - just log the best validation performance"""
        
        # Log final performance
        with open(self.log_path, 'a') as f:
            f.write(f"\nBest Model Validation Performance:\n")
            f.write(f"Best Validation Wasserstein (normalized): {self.best_wasserstein:.6f}\n")
            f.write(f"Best Validation KL: {self.best_kl:.6f}\n")

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            # Check if we're using a learning rate scheduler
            current_lr = self.model.policy.optimizer.param_groups[0]['lr']
            initial_lr = self.model.learning_rate
            
            # If we're using a scheduler, the current LR will differ from initial
            if isinstance(initial_lr, float) and current_lr != initial_lr:
                print(f"Step {self.n_calls}, Current learning rate: {current_lr}")
            elif callable(initial_lr):  # If we're using a callable LR function
                print(f"Step {self.n_calls}, Current learning rate: {current_lr}")
        
        return True

    def _generate_samples(self, num_samples):
        """Generate samples using the current model."""
        self.model.policy.set_training_mode(False)
        
        start_tokens = np.random.choice(
            np.arange(self.vocab_size), 
            size=num_samples, 
            p=self.start_token_distribution
        )

        obs = start_tokens
        lstm_states = None
        episode_starts = np.ones((num_samples,), dtype=bool)
        
        # Initialize all sequences with start token
        sequences = [[] for _ in range(num_samples)]

        # Generate full sequence_length tokens
        for _ in range(self.sequence_length):
            actions, lstm_states = self.model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
            for i, action in enumerate(actions):
                sequences[i].append(int(action))
            obs = actions
            episode_starts = np.zeros((num_samples,), dtype=bool)
        
        # Convert to tensor
        return th.tensor(sequences, dtype=th.long, device=self.discriminator.device)

    def _generate_samples_with_length(self, num_samples, seq_length):
        """Generate samples using the current model with specific length."""
        self.model.policy.set_training_mode(False)
        
        start_tokens = np.random.choice(
            np.arange(self.vocab_size), 
            size=num_samples, 
            p=self.start_token_distribution
        )

        obs = start_tokens
        lstm_states = None
        episode_starts = np.ones((num_samples,), dtype=bool)
        
        # Initialize all sequences with start token
        sequences = [[] for _ in range(num_samples)]

        # Generate tokens up to specified sequence length
        for _ in range(seq_length):
            actions, lstm_states = self.model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
            for i, action in enumerate(actions):
                sequences[i].append(int(action))
            obs = actions
            episode_starts = np.zeros((num_samples,), dtype=bool)
        
        # Convert to tensor
        return th.tensor(sequences, dtype=th.long, device=self.discriminator.device)

    def _train_discriminator(self):
        """Train the discriminator on a limited number of batches."""
        self.discriminator.train()
        
        # Calculate how many samples we need
        total_samples_needed = self.d_batch_size * self.d_epochs
        
        # Generate negative samples
        negative_samples = self._generate_samples(total_samples_needed)
        
        # Select a random subset of positive samples of the same size
        indices = th.randperm(len(self.train_data))[:total_samples_needed]
        positive_subset = self.train_data[indices]
        
        # Create balanced data loaders
        pos_loader = DataLoader(TensorDataset(positive_subset), batch_size=self.d_batch_size, shuffle=True)
        neg_loader = DataLoader(TensorDataset(negative_samples), batch_size=self.d_batch_size, shuffle=True)
        
        d_losses = []
        
        # Train for exactly d_epochs batches
        for pos_batch, neg_batch in zip(pos_loader, neg_loader):
            pos_batch = pos_batch[0]
            neg_batch = neg_batch[0]
            
            # Train discriminator on this batch
            d_loss = self.discriminator.train_step(pos_batch, neg_batch, self.d_optimizer)
            d_losses.append(d_loss)
            
            # Break if we've done enough epochs
            if len(d_losses) >= self.d_epochs:
                break
        
        # Average discriminator loss
        avg_d_loss = sum(d_losses) / len(d_losses) if d_losses else 0

        self.discriminator.eval()
        return avg_d_loss
    
    def _evaluate_discriminator(self, generated_samples):
        """Evaluate discriminator performance."""
        self.discriminator.eval()

        with th.no_grad():
            # Use validation data for evaluation
            real_preds = self.discriminator.get_sequence_probability(self.val_data)
            fake_preds = self.discriminator.get_sequence_probability(generated_samples)

            # Calculate metrics
            real_correct = (real_preds >= 0.5).sum().item()
            fake_correct = (fake_preds < 0.5).sum().item()
            
            total_samples = len(self.val_data)
            accuracy = (real_correct + fake_correct) / (2 * total_samples)
            real_prob = real_preds.mean().item()
            fake_prob = fake_preds.mean().item()
        
        metrics = {
            'accuracy': accuracy,
            'real_prob': real_prob,
            'fake_prob': fake_prob
        }

        return metrics
    
    def _calculate_metrics(self, real_data, generated_data, seq_length, stds):
        """Calculate both raw and normalized Wasserstein distances plus KL divergence."""
        
        raw_wasserstein_distances = []
        norm_wasserstein_distances = []
        kl_divergences = []
        
        # Calculate metrics for each timestep
        for t in range(seq_length):
            # Get data for this timestep
            real_t = real_data[:, t]
            gen_t = generated_data[:, t]
            
            # Calculate raw Wasserstein distance
            w_dist = wasserstein_distance(real_t, gen_t)
            raw_wasserstein_distances.append(w_dist)
            
            # Calculate normalized Wasserstein distance using the provided stds
            real_std = stds[t]
            norm_w_dist = w_dist / real_std if real_std > 0 else float('inf')
            norm_wasserstein_distances.append(norm_w_dist)
            
            # Calculate KL divergence
            vocab_size = self.vocab_size
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
            
            # Calculate KL divergence with error handling
            try:
                kl = np.sum(kl_div(real_hist, gen_hist))
                if np.isnan(kl) or np.isinf(kl):
                    print(f"Warning: KL divergence calculation at timestep {t} resulted in {kl}. Using fallback value.")
                    kl = 1000.0  # Use a high but finite value
            except Exception as e:
                print(f"Error calculating KL divergence at timestep {t}: {e}")
                kl = 1000.0
                
            kl_divergences.append(kl)
        
        # Return average metrics
        return np.mean(raw_wasserstein_distances), np.mean(norm_wasserstein_distances), np.mean(kl_divergences)
        
    def _calculate_average_reward(self):
        """Calculate average reward per episode from the rollout buffer."""
        avg_reward = 0
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            rewards = self.model.rollout_buffer.rewards
            episode_starts = self.model.rollout_buffer.episode_starts
            
            # Find episode start indices
            ep_start_idx = np.where(episode_starts)[0]
            if not episode_starts[0]:
                ep_start_idx = np.r_[0, ep_start_idx]
            
            # Calculate sum of rewards for each sequence
            if len(ep_start_idx) > 1:
                # Get the end indices for each episode
                ep_end_idx = np.r_[ep_start_idx[1:], len(rewards)]
                
                # Calculate sum of rewards for each complete sequence
                sequence_total_rewards = [np.sum(rewards[start:end]) 
                                        for start, end in zip(ep_start_idx, ep_end_idx)]
                
                # Average of the sequence totals
                avg_reward = float(np.mean(sequence_total_rewards))
            
            # If only one episode, sum all reward
            elif len(rewards) > 0:
                avg_reward = float(np.sum(rewards))
                
        return avg_reward