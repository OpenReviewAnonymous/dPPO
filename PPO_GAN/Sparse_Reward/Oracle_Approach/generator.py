import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    
    def __init__(self, vocab_size, hidden_dim, sequence_length, start_token, num_layers, device):
        
        super(Generator, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)    # Action head

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
        logits = self.output_layer(lstm_out)             # Output layer
        
        return logits, hidden
    
    def generate(self, num_samples):
        
        with th.no_grad():
            # Start token for all sequences
            x = th.full((num_samples, 1), self.start_token, dtype=th.long, device=self.device)
            hidden = None  # Let PyTorch initialize the hidden state

            generated_sequences = th.zeros(num_samples, self.sequence_length, dtype=th.long, device=self.device)

            for i in range(self.sequence_length):
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


def pretrain_generator(target_lstm, generator, optimizer, pre_epoch_num, batch_size, 
                    generated_num, positive_samples, eval_freq, 
                    lr_patience, lr_decay, log_path):
    
    print('Start pre-training...')

    # Open log file
    log = open(log_path, 'w')
    log.write('pre-training...\n')

    # For learning rate scheduling
    best_loss = float('inf')
    patience_counter = 0

    # Generate Oracle Data
    oracle_data = positive_samples
    
    # Create DataLoader
    oracle_dataset = th.utils.data.TensorDataset(oracle_data)
    oracle_loader = th.utils.data.DataLoader(oracle_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(pre_epoch_num):

        epoch_loss = 0
        batch_count = 0

        # Evaluate using the oracle every eval_freq epochs
        if epoch % eval_freq == 0 or epoch == pre_epoch_num - 1:

            generated_samples = generator.generate(generated_num)
            
            # Calculate NLL using the oracle
            nll = target_lstm.calculate_nll(generated_samples)

            # Log to file
            buffer = f'epoch:\t{epoch}\tnll:\t{nll:.4f}\n'
            log.write(buffer)
            log.flush()  # Ensure it's written immediately
        
        # Train on all batches
        for batch_data in oracle_loader:
            x = batch_data[0].to(generator.device)
            loss = generator.pretrain_step(x, optimizer)
            epoch_loss += loss
            batch_count += 1
        
        # Calculate average loss for this epoch
        avg_loss = epoch_loss / batch_count

        # Learning rate scheduling
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= lr_patience:
            # Reduce learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            print(f"Learning rate reduced to {optimizer.param_groups[0]['lr']}")
            patience_counter = 0

    log.close()
    
    print('Pretraining finished!')

def generator_adversarial_update(generator, sequences, rewards, optimizer):

    optimizer.zero_grad()

    inputs = sequences[:, :-1]  # all but the last token
    targets = sequences[:, 1:]  # all but the first token
    logits, _ = generator(inputs)
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Create a one-hot representation of the targets
    one_hot_targets = F.one_hot(targets, num_classes=generator.vocab_size).float()
    
    # Calculate the log probability of the selected actions
    # This gives us a tensor of shape [batch_size, seq_length-1, vocab_size]
    selected_log_probs = th.sum(log_probs * one_hot_targets, dim=-1)
    
    # Slice rewards to match (exclude reward for the start token)
    sequence_rewards = rewards[:, 1:]
    
    # Policy gradient loss: negative mean of (log_prob * reward)
    # We use negative because we're minimizing loss but want to maximize reward
    loss = -th.mean(selected_log_probs * sequence_rewards)
    
    # Backpropagate and update
    loss.backward()
    optimizer.step()
    
    return loss.item()
