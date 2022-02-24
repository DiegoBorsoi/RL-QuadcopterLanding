from typing import List
from typing import Optional
from typing import Tuple

from tensorflow import keras
from tensorflow import Tensor


class ActorModel(keras.Model):
    """Class containining separate network for Actor NN model."""

    def __init__(
            self,
            n_states: int,
            n_actions: int,
            hidden_layer_sizes: List[int]) -> None:
        """Initialize ActorModel class."""
        super().__init__()
        self._layer_hidden_0 = keras.layers.Dense(
            hidden_layer_sizes[0], input_shape=(n_states,), activation="relu")
        self._layer_hidden_1 = keras.layers.Dense(
            hidden_layer_sizes[1], activation="relu")
        self._layer_output = keras.layers.Dense(n_actions)

    def call(
            self,
            inputs: Tensor,
            training: Optional[bool] = None,
            mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass of the Actor neural network."""
        x = self._layer_hidden_0(inputs)
        x = self._layer_hidden_1(x)
        return self._layer_output(x)


class CriticModel(keras.Model):
    """Class containining separate network for Critic NN model."""

    def __init__(
            self,
            n_states: int,
            hidden_layer_sizes: List[int]) -> None:
        """Initialize CriticModel class."""
        super().__init__()
        self._layer_hidden_0 = keras.layers.Dense(
            hidden_layer_sizes[0], input_shape=(n_states,), activation="relu")
        self._layer_hidden_1 = keras.layers.Dense(
            hidden_layer_sizes[1], activation="relu")
        self._layer_output = keras.layers.Dense(1)

    def call(
            self,
            inputs: Tensor,
            training: Optional[bool] = None,
            mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass of the Critic neural network."""
        x = self._layer_hidden_0(inputs)
        x = self._layer_hidden_1(x)
        return self._layer_output(x)
