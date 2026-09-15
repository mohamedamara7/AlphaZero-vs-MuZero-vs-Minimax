import numpy as np
from abc import ABC, abstractmethod

class Game(ABC):
    """Base class for deterministic, two-player, zero-sum games."""
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def legal_actions(self):
        pass
    
    @abstractmethod
    def clone(self):
        pass
    
    @abstractmethod
    def render(self):
        pass
    
    @abstractmethod
    def current_player(self):
        pass
    
    @abstractmethod
    def action_size(self):
        pass
    
    @abstractmethod
    def observation_shape(self):
        pass
    
    @abstractmethod
    def result(self):
        pass    
    
    @abstractmethod
    def hash_state(self):
        pass