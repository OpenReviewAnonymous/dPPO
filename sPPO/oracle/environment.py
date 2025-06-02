import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
import torch.nn.functional as F

class TokenGenerationEnv(gym.Env):

    def __init__(self, discriminator, vocab_size, seq_length, start_token, device):
        super(TokenGenerationEnv, self).__init__()
        
        self.discriminator = discriminator
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.start_token = start_token
        self.device = device
        
        self.action_space = spaces.Discrete(vocab_size)
        self.observation_space = spaces.Discrete(vocab_size)
        
        self.current_sequence = None
        self.current_position = None
    
    def reset(self, seed=None, options=None):

        self.current_sequence = []
        self.current_position = 0
        
        return int(self.start_token), {}

    def step(self, action):
        
        self.current_sequence.append(int(action))
        self.current_position += 1
        
        # Check if the sequence is complete
        done = (self.current_position >= self.seq_length)
        
        # Calculate reward - only at the end of the sequence
        reward = 0.0
        if done:
            reward = self._get_reward()
            
        # Return observation, reward, done flag, truncated and info
        return action, reward, done, False, {}
    
    def _get_reward(self):
        """Calculate reward at the end of the sequence."""
        
        # Convert sequence to tensor
        sequence_tensor = th.tensor([self.current_sequence], dtype=th.long, device=self.device)
        
        with th.no_grad():
            
            logits = self.discriminator(sequence_tensor)
            
            target_real = th.ones_like(logits, device=self.device)

            reward = -F.binary_cross_entropy_with_logits(logits, target_real).item()
        
        return float(reward)