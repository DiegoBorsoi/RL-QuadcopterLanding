"""Paramters for use by the worker.

Written by: Zahi Kakish (zmk5)
"""
from typing import Optional
from typing import Union

from dataclasses import dataclass

import numpy as np


@dataclass
class WorkerParameters():
    """Mean Field experiement parameters stored in a dataclass."""

    __slots__ = [
        'n_states',
        'n_actions',
        'n_iterations',
        'n_episodes',
        'alpha',
        'gamma',
        'epsilon',
        'decay_rate',
        'training_delay',
        'batch_size'
    ]

    n_states: int        # default: 5
    n_actions: int       # default: 5
    n_iterations: int    # default: 1000
    n_episodes: int      # default: 5000
    alpha: float         # default: 0.3
    gamma: float         # default: 0.99
    epsilon: float       # default: 1.0
    decay_rate: int      # default: 500
    training_delay: int  # default: 100
    batch_size: int      # default: 16

    def decay_epsilon(self, episode: int, rate_value: float = 0.1) -> None:
        """Decay value of epsilon for use during training."""
        if (episode + 1) % self.decay_rate == 0:
            # The decay rate_value is same as final epsilon value.
            if self.epsilon > rate_value:
                self.epsilon -= rate_value
            else:
                self.epsilon = rate_value

    def reset_epsilon(self, new_epsilon: Optional[float] = None) -> None:
        """Reset the epsilon value."""
        if new_epsilon is None:
            self.epsilon = 1.0

        else:
            self.epsilon = new_epsilon


@dataclass
class Experience():
    """Experience undergone by the agent for an experiment."""

    __slots__ = [
        'step',
        'state',
        'action',
        'reward',
        'next_state',
        'next_action',
        'done',
    ]

    step: int
    state: np.ndarray
    action: Union[int, float]
    reward: Union[int, float]
    next_state: np.ndarray
    next_action: Union[int, float]
    done: int

    def __init__(self, n_states: int):
        """Instantiate Experience class."""
        self.step = 0
        self.state = np.empty((1, n_states))
        self.action = 0
        self.reward = 0
        self.next_state = np.empty((1, n_states))
        self.next_action = 0
        self.done = 0
    
    def new_experience(self, step: int, state: np.ndarray, action: Union[int, float], reward: Union[int,float], next_state: np.ndarray, next_action: Union[int,float], done: int):
        self.step = step
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.next_action = next_action
        self.done = done
