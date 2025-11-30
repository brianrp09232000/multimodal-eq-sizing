import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state):
        # returns Q(s, :) of shape [batch, n_actions]
        return self.net(state)


class CQLAgent:
    def __init__(self, state_dim, n_actions=3, gamma=0.99, alpha=1.0, lr=1e-3, device="cpu"):
        self.q = QNetwork(state_dim, n_actions).to(device)
        self.q_target = QNetwork(state_dim, n_actions).to(device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.alpha = alpha
        self.device = device
        self.n_actions = n_actions

    def train_step(self, batch):
        states, actions_w, rewards, next_states, dones = [
            x.to(self.device) for x in batch
        ]

        # map weights to discrete indices
        # assumes actions_w in {0.0, 0.01, 0.02}
        action_idx = (actions_w * 100).long().clamp(0, 2)  # 0,1,2

        # Q(s, ·) and Q(s, a)
        q_values = self.q(states)                 # [B, n_actions]
        q_sa = q_values.gather(1, action_idx)     # [B, 1]

        # Target Q
        with torch.no_grad():
            next_q_values = self.q_target(next_states)    # [B, n_actions]
            next_q_max, _ = next_q_values.max(dim=1, keepdim=True)
            target = rewards + self.gamma * (1 - dones) * next_q_max

        # TD loss
        td_loss = F.mse_loss(q_sa, target)

        # CQL regularizer
        # logsumexp over actions
        logsumexp_all = torch.logsumexp(q_values, dim=1, keepdim=True)  # [B,1]
        cql1 = logsumexp_all.mean()
        cql2 = q_sa.mean()
        cql_loss = self.alpha * (cql1 - cql2)

        loss = td_loss + cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "cql_loss": float(cql_loss.item()),
        }

    def update_target(self, tau=0.005):
        # soft update
        with torch.no_grad():
            for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
                p_targ.data.mul_(1 - tau).add_(tau * p.data)

    def act_greedy(self, state_np):
        """
        Given a single state np.array, return best bucket index (0,1,2).
        """
        state_t = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q(state_t)  # [1,3]
        return int(q_vals.argmax(dim=1).item())
