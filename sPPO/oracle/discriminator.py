import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Discriminator(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, num_layers, device):
        
        super(Discriminator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
        self.device = device
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM processing - get final hidden state
        _, (hidden, _) = self.lstm(embedded)
        
        # Take the final hidden state from the last layer
        final_hidden = hidden[-1]
        
        # Pass through the linear layer
        logits = self.fc(final_hidden)
        
        return logits.squeeze(-1)  # Return shape: [batch_size]
    
    def forward_step(self, x, hidden_state):
        """Process a single token step, maintaining hidden state."""
        
        # Embedding for single token
        embedded = self.embedding(x)  # Shape: [batch_size, 1, embedding_dim]
        
        # LSTM processing with hidden state
        output, hidden_state = self.lstm(embedded, hidden_state)
        
        # Extract the last layer's hidden state from the tuple
        hidden, cell = hidden_state
        final_hidden = hidden[-1]  # Last layer's hidden state
        
        # Generate probability through output layer
        logits = self.fc(final_hidden)
        
        return logits.squeeze(-1), hidden_state  # Return logits and updated hidden state

    def get_sequence_probability(self, x):

        with th.no_grad():
            logits = self.forward(x)
            rewards = th.sigmoid(logits)     # Convert to probabilities
            return rewards
    
    def train_step(self, real_data, generated_data, optimizer):

        optimizer.zero_grad()
        
        # Prepare inputs and targets
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        generated_data = generated_data.to(self.device)
        inputs = th.cat([real_data, generated_data], dim=0)

        smooth_real = 0.9  # Instead of 1.0
        smooth_fake = 0.1  # Instead of 0.0
    
        targets = th.cat([
            th.ones(batch_size, device=self.device) * smooth_real,  # Smoothed real targets 
            th.zeros(batch_size, device=self.device) * smooth_fake  # Smoothed fake targets
        ], dim=0)
        
        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()

def pretrain_discriminator(target_lstm, generator, discriminator, optimizer, outer_epochs, inner_epochs, batch_size, generated_num, positive_samples, log_file, lr_patience, lr_decay, min_lr):
        
    # Open log file
    log = open(log_file, 'w')
    log.write('Discriminator pre-training...\n')
    
    total_epochs = 0
    best_loss = float('inf')
    patience_counter = 0

    # Outer loop
    for outer_epoch in range(outer_epochs):
            
        # Generate new negative samples for each outer epoch
        generator.eval()
        with th.no_grad():
            negative_samples = generator.generate(generated_num)

        pos_loader = DataLoader(TensorDataset(positive_samples), batch_size=batch_size, shuffle=True)   
        neg_loader = DataLoader(TensorDataset(negative_samples), batch_size=batch_size, shuffle=True)
        
        epoch_total_loss = 0
        epoch_batches = 0

        for inner_epoch in range(inner_epochs):
            
            # Set discriminator to training mode
            discriminator.train()
            
            total_loss = 0
            num_batches = 0
            
            # Iterate through batches
            for (pos_batch,), (neg_batch,) in zip(pos_loader, neg_loader):
                
                pos_batch = pos_batch.to(discriminator.device)
                neg_batch = neg_batch.to(discriminator.device)
                
                # Use the train_step method which handles the full training loop
                loss = discriminator.train_step(pos_batch, neg_batch, optimizer)
                
                total_loss += loss
                num_batches += 1
                
                epoch_total_loss += loss
                epoch_batches += 1
            
            total_epochs += 1
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            # Evaluation
            eval_metrics = evaluate_discriminator(discriminator, target_lstm, generator, num_samples=int(generated_num/5))
            
            log_str = f'epoch:\t{total_epochs}\tloss:\t{avg_loss:.4f}\t'
            log_str += f'accuracy:\t{eval_metrics["accuracy"]:.4f}\t'
            log_str += f'real_prob\t{eval_metrics["real_prob"]:.4f}\tfake_prob\t{eval_metrics["fake_prob"]:.4f}'
            
            log.write(log_str + '\n')
            log.flush()
        
        # Learning rate scheduling
        avg_epoch_loss = epoch_total_loss / epoch_batches if epoch_batches > 0 else float('inf')
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= lr_patience:
            # Reduce learning rate, but don't go below min_lr
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                new_lr = max(current_lr * lr_decay, min_lr)
                param_group['lr'] = new_lr
            
            # Log the learning rate change
            current_lr = optimizer.param_groups[0]['lr']
            log.write(f"Learning rate reduced to {current_lr}\n")
            log.flush()
            
            # Only reset patience counter if we actually changed the learning rate
            if current_lr > min_lr:
                patience_counter = 0
            else:
                log.write("Minimum learning rate reached.\n")
                log.flush()
    
    log.close()

def evaluate_discriminator(discriminator, target_lstm, generator, num_samples):
    
    discriminator.eval()
    target_lstm.eval()
    generator.eval()
        
    with th.no_grad():
        # Generate data
        real_data = target_lstm.generate(num_samples)
        fake_data = generator.generate(num_samples)
        
        # Get predictions - using get_sequence_probability instead of get_reward
        real_preds = discriminator.get_sequence_probability(real_data)
        fake_preds = discriminator.get_sequence_probability(fake_data)
        
        # Calculate metrics
        real_correct = (real_preds >= 0.5).sum().item()
        fake_correct = (fake_preds < 0.5).sum().item()
        
        accuracy = (real_correct + fake_correct) / (2 * num_samples)
        real_prob = real_preds.mean().item()
        fake_prob = fake_preds.mean().item()
    
    metrics = {
        'accuracy': accuracy,
        'real_prob': real_prob,
        'fake_prob': fake_prob
    }
    
    return metrics