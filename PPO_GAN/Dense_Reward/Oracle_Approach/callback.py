import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    
    def __init__(self, discriminator, oracle, d_optimizer, 
                 d_steps, k_epochs, d_batch_size,
                 positive_samples, sequence_length, start_token, generated_num,
                 eval_freq=1, verbose=0, log_path=None):
        
        super(CustomCallback, self).__init__(verbose)
        
        # Models and optimizer
        self.discriminator = discriminator
        self.oracle = oracle
        self.oracle.eval()
        self.d_optimizer = d_optimizer
        
        # Training parameters
        self.d_steps = d_steps
        self.k_epochs = k_epochs
        self.d_batch_size = d_batch_size
        self.generated_num = generated_num
        self.eval_freq = eval_freq
        
        # Data
        self.positive_samples = positive_samples
        self.num_samples = len(positive_samples)
        self.start_token = start_token
        self.sequence_length = sequence_length
        
        self.rollout_count = 0

        self.log_path = log_path if log_path else '0_ppo_sparse_training.txt'
        
    def _on_training_start(self):

        """Called at the start of training"""
        pass
        
    def _on_rollout_start(self):
        """Called at the start of a rollout"""
        pass
    
    def _on_rollout_end(self):
        """Called at the end of a rollout - this is where we'll train the discriminator"""

        self.rollout_count += 1

        negative_samples = self._generate_samples(self.generated_num)

        # Train the discriminator
        d_loss = self._train_discriminator(negative_samples)

        # Only evaluate and log after eval_freq rollouts
        if self.rollout_count % self.eval_freq == 0:
            # Evaluation
            nll = self.oracle.calculate_nll(negative_samples)
            disc_metrics = self._evaluate_discriminator(negative_samples)

            # Get current PPO metrics directly from logger
            policy_loss = self.logger.name_to_value.get('train/policy_gradient_loss', 0)
            value_loss = self.logger.name_to_value.get('train/value_loss', 0)
            entropy = self.logger.name_to_value.get('train/entropy_loss', 0)

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

            # Log to file after each rollout
            with open(self.log_path, 'a') as f:
                # Header
                if self.rollout_count == self.eval_freq:   
                    f.write('rollout\tnll\tpolicy_loss\tvalue_loss\tentropy\td_loss\td_accuracy\treal_prob\tfake_prob\tavg_reward\n')
                f.write(f'{self.rollout_count}\t{nll:.6f}\t{policy_loss:.6f}\t{value_loss:.6f}\t{entropy:.6f}\t'
                        f'{d_loss:.6f}\t{disc_metrics["accuracy"]:.6f}\t{disc_metrics["real_prob"]:.6f}\t{disc_metrics["fake_prob"]:.6f}\t{avg_reward:.6f}\n')
                f.flush()
                    
    def _on_training_end(self):
        """Called at the end of training"""
        pass

    def _on_step(self) -> bool:
    
        if self.n_calls % 100 == 0:
            # Check if we're using a learning rate scheduler
            # Access optimizer through the policy
            current_lr = self.model.policy.optimizer.param_groups[0]['lr']
            initial_lr = self.model.learning_rate
            
            # If we're using a scheduler, the current LR will differ from initial
            if isinstance(initial_lr, float) and current_lr != initial_lr:
                print(f"Step {self.n_calls}, Current learning rate: {current_lr}")
            elif callable(initial_lr):  # If we're using a callable LR function
                print(f"Step {self.n_calls}, Current learning rate: {current_lr}")
        
        return True

    def _generate_samples(self, num_samples):

        self.model.policy.set_training_mode(False)
        obs = np.array([self.start_token] * num_samples)
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
        
    def _train_discriminator(self, negative_samples):
        
        self.discriminator.train()
        
        d_losses = []
        
        for _ in range(self.d_steps):
            
            # Create data loaders
            pos_loader = DataLoader(TensorDataset(self.positive_samples), batch_size=self.d_batch_size, shuffle=True)
            neg_loader = DataLoader(TensorDataset(negative_samples), batch_size=self.d_batch_size, shuffle=True)
            
            # Train discriminator for k epochs
            for _ in range(self.k_epochs):
                
                batch_d_losses = []
                for (pos_batch,), (neg_batch,) in zip(pos_loader, neg_loader):
                    d_loss = self.discriminator.train_step(pos_batch, neg_batch, self.d_optimizer)
                    batch_d_losses.append(d_loss)
                
                if batch_d_losses: d_losses.append(sum(batch_d_losses) / len(batch_d_losses))
        
        # Average discriminator loss
        avg_d_loss = sum(d_losses) / len(d_losses) if d_losses else 0

        self.discriminator.eval()
        
        return avg_d_loss
    
    def _evaluate_discriminator(self, negative_samples):
    
        self.discriminator.eval()

        with th.no_grad():
            real_preds = self.discriminator.get_sequence_probability(self.positive_samples)
            fake_preds = self.discriminator.get_sequence_probability(negative_samples)

            # Calculate metrics
            real_correct = (real_preds >= 0.5).sum().item()
            fake_correct = (fake_preds < 0.5).sum().item()
            
            accuracy = (real_correct + fake_correct) / (2 * self.num_samples)
            real_prob = real_preds.mean().item()
            fake_prob = fake_preds.mean().item()
        
        metrics = {
            'accuracy': accuracy,
            'real_prob': real_prob,
            'fake_prob': fake_prob
        }
    
        return metrics
