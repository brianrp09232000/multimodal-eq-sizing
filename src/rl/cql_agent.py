from dataclasses import dataclass
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_q_network(
    state_dim: int,
    n_actions: int = 3,
    hidden_dim: int = 128,
) -> keras.Model:
    """
    Simple MLP that maps state -> Q(s, ·) over discrete actions.
    """
    inputs = keras.Input(shape=(state_dim,), name="state")
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dense(hidden_dim, activation="relu")(x)
    outputs = layers.Dense(n_actions, name="q_values")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="q_network")
    return model


@dataclass
class CQLConfig:
    gamma: float = 0.99
    alpha: float = 1.0        # CQL regularization strength
    lr: float = 1e-3
    tau: float = 0.005        # target network soft-update rate
    n_actions: int = 3
    hidden_dim: int = 128


class CQLAgentTF:
    """
    Conservative Q-Learning agent with discrete actions in TensorFlow/Keras.

    Q-network outputs Q(s, a) for all a in {0, ..., n_actions-1}.
    """

    def __init__(self, state_dim: int, config: CQLConfig | None = None):
        if config is None:
            config = CQLConfig()

        self.config = config

        # Main and target Q networks
        self.q = build_q_network(
            state_dim=state_dim,
            n_actions=config.n_actions,
            hidden_dim=config.hidden_dim,
        )
        self.q_target = build_q_network(
            state_dim=state_dim,
            n_actions=config.n_actions,
            hidden_dim=config.hidden_dim,
        )
        self.q_target.set_weights(self.q.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=config.lr)

    def _gather_q_sa(self, q_values: tf.Tensor, action_idx: tf.Tensor) -> tf.Tensor:
        """
        Given Q(s, ·) [B, A] and action indices [B, 1] or [B],
        gather Q(s, a) for the chosen action.
        """
        if len(action_idx.shape) == 2:
            action_idx = tf.squeeze(action_idx, axis=-1)  # [B]

        batch_size = tf.shape(q_values)[0]
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        indices = tf.stack([batch_indices, action_idx], axis=1)  # [B, 2]
        return tf.gather_nd(q_values, indices)[:, tf.newaxis]   # [B, 1]

    @tf.function
    def train_step(
        self,
        states: tf.Tensor,
        actions_idx: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """
        One CQL training step on a batch.

        states:      [B, state_dim]
        actions_idx: [B, 1] or [B], discrete indices {0,1,2}
        rewards:     [B, 1]
        next_states: [B, state_dim]
        dones:       [B, 1] (1.0 if terminal, else 0.0)
        """
        gamma = self.config.gamma
        alpha = self.config.alpha

        with tf.GradientTape() as tape:
            # Q(s, ·) and Q(s, a)
            q_values = self.q(states)  # [B, n_actions]
            q_sa = self._gather_q_sa(q_values, actions_idx)  # [B, 1]

            # Target Q(s', ·)
            next_q_values = self.q_target(next_states)       # [B, n_actions]
            next_q_max = tf.reduce_max(next_q_values, axis=1, keepdims=True)  # [B, 1]

            # y = r + gamma * (1 - done) * max_a' Q_target(s', a')
            target = rewards + gamma * (1.0 - dones) * next_q_max

            # TD loss
            td_loss = tf.reduce_mean(tf.square(q_sa - tf.stop_gradient(target)))

            # CQL penalty: alpha * (E_s[logsumexp Q(s, a)] - E_{(s,a)~D}[Q(s,a)])
            logsumexp_all = tf.reduce_logsumexp(q_values, axis=1, keepdims=True)  # [B,1]
            cql1 = tf.reduce_mean(logsumexp_all)
            cql2 = tf.reduce_mean(q_sa)
            cql_loss = alpha * (cql1 - cql2)

            loss = td_loss + cql_loss

        grads = tape.gradient(loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q.trainable_variables))

        return {
            "loss": loss,
            "td_loss": td_loss,
            "cql_loss": cql_loss,
        }

    def update_target(self):
        """
        Soft-update target network parameters:
          θ_target ← τ θ + (1 - τ) θ_target
        """
        tau = self.config.tau
        q_weights = self.q.get_weights()
        target_weights = self.q_target.get_weights()
        new_weights = []
        for w, w_t in zip(q_weights, target_weights):
            new_weights.append(tau * w + (1.0 - tau) * w_t)
        self.q_target.set_weights(new_weights)

    def act_greedy(self, state_np: np.ndarray) -> int:
        """
        Given a single state as a numpy array of shape [state_dim],
        return the best discrete action index (0, 1, 2).
        """
        state_batch = np.expand_dims(state_np, axis=0)  # [1, state_dim]
        q_vals = self.q(state_batch, training=False).numpy()[0]  # [n_actions]
        return int(np.argmax(q_vals))
