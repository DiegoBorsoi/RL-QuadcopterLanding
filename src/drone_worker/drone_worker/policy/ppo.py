from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from drone_worker.policy.models import ActorModel
from drone_worker.policy.models import CriticModel


STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3
NEXT_ACTION = 4
DONE = 5

class WorkerPPO():
    """PPO Class containing all relatvent RL information."""

    def __init__(
            self,
            n_states: int,
            n_actions: int,
            alpha: float,
            gamma: float,
            hidden_layer_sizes: List[int],
            use_gpu: bool = False) -> None:
        """Initialize the PPO class."""
        self._n_states = n_states
        self._n_actions = n_actions
        self._alpha: float = alpha
        self._gamma: float = gamma
        self._hidden_layer_sizes = hidden_layer_sizes

        # Turn off GPU, if needed.
        if not use_gpu:
            tf.config.set_visible_devices([], 'GPU')

        # Use Actor-Critic model instead of that within WorkerPolicyREINFORCE.
        self._neural_net = ActorModel(n_states, n_actions, hidden_layer_sizes)
        self._critic_net = CriticModel(n_states, hidden_layer_sizes)
        
        # Set tensorflow loss optimizer
        self._actor_optimizer = Adam(learning_rate=self._alpha, clipvalue=1.0)
        self._critic_optimizer = Adam(learning_rate=self._alpha, clipvalue=1.0)
        
        # MSE loss
        self._loss_function = keras.losses.MeanSquaredError(
            reduction=keras.losses.Reduction.SUM)

        # Build and compile Actor-Critic model
        self._neural_net.build((1, n_states))
        self._critic_net.build((1, n_states))

        # Create additional variable for gradients.
        self._actor_gradients: List[np.ndarray] = []
        self._critic_gradients: List[np.ndarray] = []

        self.clip_param = 0.2


    @property
    def atype(self) -> str:
        """Return type of RL algorithm as string."""
        return 'PPO'

    def train(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int = 16) -> None:
        """Train the soft actor-critic policy based on a sample batch."""
        values_pred = self._critic_net(batch[STATE])
        next_values_pred = self._critic_net(batch[NEXT_STATE])
        #returns = self.calculate_nstep_returns(batch, batch_size, next_values_pred)
        returns = self.calculate_gae_returns(batch, batch_size, values_pred, next_values_pred)
        # Compute the action probs and value for current and next state.
        action_logits = self._neural_net(batch[STATE])
        action_probs = tf.nn.softmax(action_logits)

        for _ in range(10):
            c_loss = self.train_critic(returns, batch, batch_size)
            self.train_actor(returns, values_pred, action_probs, c_loss, batch, batch_size)
            self.optimize

    def optimize(self) -> None:
        """Optimize global network policy."""
        self._actor_optimizer.apply_gradients(
                zip(self._actor_gradients, self._neural_net.trainable_variables))

        self._critic_optimizer.apply_gradients(
                zip(self._critic_gradients, self._critic_net.trainable_variables))

    def train_actor(
            self,
            returns: np.ndarray,
            values: np.ndarray,
            action_probs: np.ndarray,
            c_loss: tf.Tensor,
            batch: Tuple[np.ndarray],
            batch_size: int = 16) -> None:
        """Train the actor policy based on a sample batch."""
        #values = self._critic_net(batch[STATE])
        with tf.GradientTape() as tape:
            # Compute the returns and loss.
            loss = self.calculate_actor_loss(
                batch, action_probs, returns, values, batch_size, c_loss)

        # Calculate and apply graidents.
        self._actor_gradients = tape.gradient(
            loss, self._neural_net.trainable_variables)

    def train_critic(
            self,
            returns: np.ndarray,
            batch: Tuple[np.ndarray],
            batch_size: int = 16) -> tf.Tensor:
        """Train the actor policy based on a sample batch."""
        _ = batch_size  # TODO: Batch size is not used.
        with tf.GradientTape() as tape:
            # Compute the value for current and next state.
            values = self._critic_net(batch[STATE])

            # Compute the returns and loss.
            # print(f'Grads: {[var.name for var in tape.watched_variables()]}')
            loss = self.calculate_critic_loss(returns, values)

        # Calculate and apply gradients.
        self._critic_gradients = tape.gradient(
            loss, self._critic_net.trainable_variables)

        return loss

    def calculate_nstep_returns(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int,
            next_v_pred: tf.Tensor) -> np.ndarray:
        """Calculate n-step advantage returns."""
        ret_value = np.zeros_like(batch[REWARD])
        # try:
        #     future_ret = next_v_pred.numpy()[-1]
        #     print(f'Future Return: {future_ret}')

        # except IndexError:
        #     future_ret = next_v_pred.numpy()
        future_ret = 0.0

        for t in reversed(range(batch_size + 1)):
            ret_value[t] = future_ret = batch[REWARD][t] + self._gamma * future_ret * (1 - batch[DONE][t])

        return ret_value

    def calculate_gae_returns(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int,
            v_preds: tf.Tensor,
            next_v_pred: tf.Tensor) -> np.ndarray:
        """Calculate Generalaized Advantage Estimation (GAE) returns."""
        gaes = np.zeros_like(batch[REWARD])
        future_gae = 0.0

        #print("Batch: %s, batch_size: %s, v_preds: %s, next_v_pred: %s" % (len(batch), batch_size, v_preds.shape, next_v_pred.shape))

        for t in reversed(range(batch_size)):
            delta = batch[REWARD][t] + self._gamma * next_v_pred[t] * (1 - batch[DONE][t]) - v_preds[t]
            future_gae = delta + self._gamma * 0.95 * (1 - batch[DONE][t]) * future_gae  # lambda = 0.95
            gaes[t] = future_gae + v_preds[t]

        return gaes

    def calculate_actor_loss(
            self,
            batch: Tuple[np.ndarray],
            action_probs: Union[np.ndarray, tf.Tensor],
            returns: Union[np.ndarray, tf.Tensor],
            values: Union[np.ndarray, tf.Tensor],
            batch_size: int,
            c_loss: tf.Tensor) -> tf.Tensor:
        """Calculate the Actor network loss."""
        advantage = returns - values
        # TODO: adjust on mean and std

        #action_log_probs = tf.math.log(action_probs)
        #idx = tf.Variable(
        #    np.append(np.arange(batch_size + 1).reshape(batch_size + 1, 1), action_batch, axis=1),
        #    dtype=tf.int32
        #)
        #act_log_probs = tf.reshape(tf.gather_nd(action_log_probs, idx), (batch_size + 1, 1))

        new_action_logits = self._neural_net(batch[STATE])
        new_action_probs = tf.nn.softmax(new_action_logits)

        # Actor Loss with Entropy
        entropy = np.sum(-1 * new_action_probs.numpy() * np.log(new_action_probs.numpy()), axis=1)
        # print(f'Entropy: {entropy.mean()}')

        ratio = tf.math.divide(new_action_probs, action_probs)

        s1 = ratio * advantage
        s2 = np.clip(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

        actor_loss = -1 * tf.math.reduce_mean(tf.math.minimum(s1, s2)) + c_loss - (0.001 * entropy.mean())

        # entropy = np.sum(-1 * action_probs * np.log(action_probs), axis=1).reshape(batch_size + 1, 1)
        # actor_loss = -1 * tf.math.reduce_sum((act_log_probs * advantage) - (0.01 * entropy))

        #actor_loss = -1 * tf.math.reduce_sum(act_log_probs * advantage)
        # print(f'Actor Loss: {actor_loss}')

        return actor_loss

    def calculate_critic_loss(
            self,
            returns: Union[np.ndarray, tf.Tensor],
            values: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        """Calculate the Critic network loss."""
        critic_loss = 0.5 * self._loss_function(returns, values)
        # print(f'Critic Loss: {critic_loss}')

        return critic_loss

    def act(
            self,
            state: np.ndarray,
            epsilon: Optional[float] = None) -> Union[int, np.integer]:
        """Apply the policy for a ROS inference service request."""
        _ = epsilon  # Unused by REINFORCE
        prob = self._neural_net(state)
        return tf.random.categorical(prob, 1)[0, 0].numpy()

    def load_model(self, path_to_model: str) -> None:
        """Load model for inference or training use."""
        self._neural_net = keras.models.load_model(path_to_model + 'trained_model_actor')
        self._critic_net = keras.models.load_model(path_to_model + 'trained_model_critic')

    def save_model(self, path_to_model: str) -> None:
        """Load model for inference or training use."""
        # Predict required for saving due to odd error found here:
        # https://github.com/tensorflow/tensorflow/issues/31057
        
        # self._neural_net.predict(np.arange(self._n_states).reshape(1, self._n_states))
        #self._neural_net.predict(np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.92, 0.0]]))
        self._neural_net.save(path_to_model + 'trained_model_actor')

        # self._critic_net.predict(np.arange(self._n_states).reshape(1, self._n_states))
        #self._critic_net.predict(np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.92, 0.0]]))
        self._critic_net.save(path_to_model + 'trained_model_critic')