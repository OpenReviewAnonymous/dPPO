import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle

class Oracle(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, sequence_length, start_token, device):
        
        super(Oracle, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        
        self.device = device
        
        # Define layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize on device
        self.to(self.device)
       
    def forward(self, x, hidden=None):

        emb = self.embeddings(x)                    # [batch_size, sequence_length, embedding_dim]
        lstm_out, hidden = self.lstm(emb, hidden)   # lstm_out: [batch_size, sequence_length, hidden_dim]
        logits = self.output_layer(lstm_out)        # [batch_size, sequence_length, vocab_size]
        
        return logits, hidden
    
    def generate(self, num_samples):

        with th.no_grad():
            
            # Start token for all sequences
            x = th.full((num_samples, 1), self.start_token, dtype=th.long, device=self.device)
            hidden = None  # Let Pyth initialize the hidden state

            generated_sequences = th.zeros(num_samples, self.sequence_length, dtype=th.long, device=self.device)

            for i in range(self.sequence_length):
                # Forward pass
                emb = self.embeddings(x[:, -1:])  # Only use the last token
                lstm_out, hidden = self.lstm(emb, hidden)
                logits = self.output_layer(lstm_out)
                
                # Sample from distribution
                probs = F.softmax(logits.squeeze(1), dim=-1)
                next_token = th.multinomial(probs, 1)
                
                # Add to sequence
                generated_sequences[:, i] = next_token.squeeze()
                
                # Update input for next step (only need the current token, not the entire history)
                x = next_token
            
            return generated_sequences

    def calculate_nll(self, generated_sequences):

        with th.no_grad():
            # Use all tokens except the last one as input
            inputs = generated_sequences[:, :-1]
            
            # Use all tokens except the first one as targets
            targets = generated_sequences[:, 1:]
            
            # Forward pass
            logits, _ = self.forward(inputs)
            
            # Calculate negative log-likelihood
            nll = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), reduction='mean')
            
            return nll.item()

    def load_params(self, params_path):
        """
        Load parameters from a TensorFlow list format.
        """
        try:
            with open(params_path, 'rb') as f:
                try:
                    params = pickle.load(f)
                except UnicodeDecodeError:
                    f.seek(0)
                    params = pickle.load(f, encoding='latin1')
        except Exception as e:
            print(f"Error loading pickle file: {str(e)}")
            return self
        
        with th.no_grad():
            # 1. Embeddings
            self.embeddings.weight.copy_(th.tensor(params[0], dtype=th.float32))
            
            # 2. LSTM Parameters
            # Extract individual LSTM weights
            Wi, Ui, bi = params[1], params[2], params[3]  # Input gate
            Wf, Uf, bf = params[4], params[5], params[6]  # Forget gate
            Wo, Uo, bo = params[7], params[8], params[9]  # Output gate
            Wc, Uc, bc = params[10], params[11], params[12]  # Cell state
            
            # Concatenate the weights in Pyth's expected format
            weight_ih = np.vstack([Wi, Wf, Wc, Wo])
            weight_hh = np.vstack([Ui, Uf, Uc, Uo])
            
            # Bias is also concatenated
            bias_ih = np.concatenate([bi, bf, bc, bo])
            bias_hh = np.zeros_like(bias_ih)
            
            # Copy to Pyth model
            self.lstm.weight_ih_l0.copy_(th.tensor(weight_ih, dtype=th.float32))
            self.lstm.weight_hh_l0.copy_(th.tensor(weight_hh, dtype=th.float32))
            self.lstm.bias_ih_l0.copy_(th.tensor(bias_ih, dtype=th.float32))
            self.lstm.bias_hh_l0.copy_(th.tensor(bias_hh, dtype=th.float32))
            
            # 3. Output layer
            self.output_layer.weight.copy_(th.tensor(params[13].T, dtype=th.float32))
            self.output_layer.bias.copy_(th.tensor(params[14], dtype=th.float32))
        
        return self

    def save_params(self, path):
        th.save(self.state_dict(), path)
        
    def save_samples(self, samples, file_path):
        with open(file_path, 'w') as f:
            for sample in samples.cpu().numpy():
                f.write(' '.join([str(int(x)) for x in sample]) + '\n')

