import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

class Discriminator(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, device, num_layers=3):
        
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

        # Instead of hard 0/1 targets, use 0.1/0.9
        smooth_real = 0.85  # Instead of 1.0
        smooth_fake = 0.15  # Instead of 0.0
    
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


def pretrain_discriminator(generator, discriminator, optimizer, train_data, val_data, batch_size, pretrain_epochs, log_file, lr_patience, lr_decay, min_lr):
        
    # Open log file
    log = open(log_file, 'w')
    log.write('Discriminator pre-training...\n')
    
    seed_str = Path(log_file).stem.split('_')[0]  # Extract seed from log filename
    models_dir = Path(os.getenv('MODELS_DIR', './saved_models_training'))
    best_model_path = str(models_dir / f"{seed_str}_discriminator_best_model.pth")
    
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Prepare positive examples (real data)
    positive_samples = th.tensor(train_data, dtype=th.long, device=discriminator.device)
    val_positive = th.tensor(val_data, dtype=th.long, device=discriminator.device)
    
    # Training loop
    for epoch in range(pretrain_epochs):
            
        # Generate new negative samples for each epoch
        generator.eval()
        with th.no_grad():
            negative_samples = generator.generate(len(train_data))
            val_negative = generator.generate(len(val_data))

        # Create data loaders
        pos_loader = DataLoader(TensorDataset(positive_samples), batch_size=batch_size, shuffle=True)   
        neg_loader = DataLoader(TensorDataset(negative_samples), batch_size=batch_size, shuffle=True)
        
        # For validation data
        val_pos_loader = DataLoader(TensorDataset(val_positive), batch_size=batch_size, shuffle=False)
        val_neg_loader = DataLoader(TensorDataset(val_negative), batch_size=batch_size, shuffle=False)
        
        # Training phase
        discriminator.train()
        train_loss = 0
        train_batches = 0
        
        # Iterate through batches
        for (pos_batch,), (neg_batch,) in zip(pos_loader, neg_loader):
            loss = discriminator.train_step(pos_batch, neg_batch, optimizer)
            train_loss += loss
            train_batches += 1
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        
        # Validation phase
        discriminator.eval()
        val_loss = 0
        val_batches = 0
        
        with th.no_grad():
            for (pos_batch,), (neg_batch,) in zip(val_pos_loader, val_neg_loader):
                # Forward pass
                inputs = th.cat([pos_batch, neg_batch], dim=0)
                batch_size = pos_batch.size(0)
                
                # Use smoothed labels for consistency
                smooth_real = 0.9
                smooth_fake = 0.1
                targets = th.cat([
                    th.ones(batch_size, device=discriminator.device) * smooth_real,
                    th.zeros(batch_size, device=discriminator.device) * smooth_fake
                ], dim=0)
                
                logits = discriminator(inputs)
                loss = th.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        
        # Get evaluation metrics
        eval_metrics = evaluate_discriminator(discriminator, val_positive, val_negative)
        
        log_str = f'epoch:\t{epoch}\ttrain_loss:\t{avg_train_loss:.4f}\tval_loss:\t{avg_val_loss:.4f}\t'
        log_str += f'accuracy:\t{eval_metrics["accuracy"]:.4f}\t'
        log_str += f'real_prob\t{eval_metrics["real_prob"]:.4f}\tfake_prob\t{eval_metrics["fake_prob"]:.4f}'
        
        log.write(log_str + '\n')
        log.flush()
        
        # Learning rate scheduling based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model state
            best_model_state = {
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss
            }
            print(f"New best discriminator at epoch {epoch} with validation loss: {avg_val_loss:.5f}")
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
    
    # Save the best model
    #best_model_path = log_file.replace('.txt', '_best_model.pth')
    if best_model_state is not None:
        th.save(best_model_state, best_model_path)
        print(f"Best discriminator model saved to {best_model_path}")
        
        # Load the best model state
        discriminator.load_state_dict(best_model_state['model_state_dict'])
    
    log.close()
    return best_model_path

def evaluate_discriminator(discriminator, val_real, val_fake):

    discriminator.eval()
        
    with th.no_grad():
        # Get predictions
        real_preds = discriminator.get_sequence_probability(val_real)
        fake_preds = discriminator.get_sequence_probability(val_fake)
        
        # Calculate metrics
        real_correct = (real_preds >= 0.5).sum().item()
        fake_correct = (fake_preds < 0.5).sum().item()
        
        accuracy = (real_correct + fake_correct) / (len(val_real) + len(val_fake))
        real_prob = real_preds.mean().item()
        fake_prob = fake_preds.mean().item()
    
    metrics = {
        'accuracy': accuracy,
        'real_prob': real_prob,
        'fake_prob': fake_prob
    }
    
    return metrics
