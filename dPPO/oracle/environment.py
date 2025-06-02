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
        
        self.current_position = None
        self.hidden_state = None
        self.previous_loss = None
         
    def reset(self, seed=None, options=None):
        
        self.current_position = 0
        self.hidden_state = None
        self.previous_loss = None
        
        return int(self.start_token), {}

    def step(self, action):
        # Get reward for this step
        reward = self._get_step_reward(action)
        
        # Increment position after calculating reward
        self.current_position += 1
        
        # Check if the sequence is complete
        done = (self.current_position >= self.seq_length)
        
        # Return observation, reward, done flag, truncated and info
        return action, reward, done, False, {}
    
    def _get_step_reward(self, action):
        # Convert action to tensor
        action_tensor = th.tensor([[action]], dtype=th.long, device=self.device)
        
        with th.no_grad():
            # Process the current token
            logits, self.hidden_state = self.discriminator.forward_step(action_tensor, self.hidden_state)
            
            # Create a "real" target (we want the generator to produce tokens that the discriminator thinks are real)
            target_real = th.ones_like(logits, device=self.device)

            # Calculate BCE loss for current state
            current_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, target_real).item()
            
            if self.current_position == 0:
                # For the first token, use negative of raw BCE
                reward = -current_loss
            else:
                # For subsequent tokens, reward is the improvement (delta) in BCE
                # If previous_loss - current_loss is positive, it means we improved
                reward = self.previous_loss - current_loss
            
            # Update previous loss for next step
            self.previous_loss = current_loss
                
        return float(reward)

