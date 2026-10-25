"""DQN / Double DQN for victim selection — written from scratch on PyTorch tensors.

The action set changes every step (a different list of candidate blocks), so the network is a
*per-candidate scorer*:  Q(s, a_i) = f_theta([candidate features_i, global features]).
One small MLP scores every candidate; padded candidates are masked out with -inf.
This is ordinary DQN (lecture week 6) where the action enters the network as features
instead of as an output index.

Pieces (same names as the lecture slides):
  * experience replay buffer (uniform sampling)
  * target network theta' (hard copy every `target_every` updates)
  * epsilon-greedy behaviour policy, epsilon annealed linearly
  * TD target  y = r + gamma * max_a' Q_theta'(s', a')           (DQN)
               y = r + gamma * Q_theta'(s', argmax_a' Q_theta(s', a'))   (Double DQN)
  * Huber loss, Adam, gradient-norm clipping
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

NEG = -1e9


class Scorer(nn.Module):
    def __init__(self, n_cand_feat: int, n_glob: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_cand_feat + n_glob, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, cand, glob):
        # cand: (B, k, F)  glob: (B, G)  -> scores (B, k)
        g = glob.unsqueeze(1).expand(-1, cand.shape[1], -1)
        return self.net(torch.cat([cand, g], dim=-1)).squeeze(-1)


class Replay:
    def __init__(self, cap: int, k: int, F: int, G: int, seed: int):
        self.cap, self.n, self.i = cap, 0, 0
        self.cand = np.zeros((cap, k, F), np.float32)
        self.mask = np.zeros((cap, k), np.float32)
        self.glob = np.zeros((cap, G), np.float32)
        self.act = np.zeros(cap, np.int64)
        self.rew = np.zeros(cap, np.float32)
        self.ncand = np.zeros((cap, k, F), np.float32)
        self.nmask = np.zeros((cap, k), np.float32)
        self.nglob = np.zeros((cap, G), np.float32)
        self.done = np.zeros(cap, np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, o2, done):
        i = self.i
        self.cand[i], self.mask[i], self.glob[i] = o["cand"], o["mask"], o["glob"]
        self.act[i], self.rew[i], self.done[i] = a, r, float(done)
        self.ncand[i], self.nmask[i], self.nglob[i] = o2["cand"], o2["mask"], o2["glob"]
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs: int):
        idx = self.rng.integers(0, self.n, size=bs)
        t = torch.from_numpy
        return (t(self.cand[idx]), t(self.mask[idx]), t(self.glob[idx]), t(self.act[idx]),
                t(self.rew[idx]), t(self.ncand[idx]), t(self.nmask[idx]), t(self.nglob[idx]),
                t(self.done[idx]))


class DQNAgent:
    def __init__(self, k: int, F: int, G: int, gamma: float = 0.99, lr: float = 5e-4,
                 hidden: int = 64, double: bool = False, buffer: int = 100_000, batch: int = 128,
                 target_every: int = 500, train_every: int = 2, learn_start: int = 2000,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_steps: int = 50_000,
                 reward_shift: float = 0.0, seed: int = 0):
        torch.manual_seed(seed)
        self.k, self.gamma, self.double = k, gamma, double
        self.name = "double-dqn" if double else "dqn"
        self.q = Scorer(F, G, hidden)
        self.qt = Scorer(F, G, hidden)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = Replay(buffer, k, F, G, seed)
        self.batch, self.target_every, self.train_every = batch, target_every, train_every
        self.learn_start = learn_start
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_steps
        self.reward_shift = reward_shift
        self.rng = np.random.default_rng(seed + 17)
        self.steps = 0
        self.updates = 0
        self.last_loss = float("nan")

    def epsilon(self) -> float:
        f = min(1.0, self.steps / max(self.eps_decay, 1))
        return self.eps_start + f * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def scores(self, obs) -> np.ndarray:
        c = torch.from_numpy(obs["cand"]).unsqueeze(0)
        g = torch.from_numpy(obs["glob"]).unsqueeze(0)
        s = self.q(c, g)[0].numpy()
        return np.where(obs["mask"] > 0, s, NEG)

    def act(self, obs, greedy: bool = False) -> int:
        valid = np.flatnonzero(obs["mask"] > 0)
        if not greedy and self.rng.random() < self.epsilon():
            return int(valid[self.rng.integers(len(valid))])
        return int(np.argmax(self.scores(obs)))

    def observe(self, o, a, r, o2, done):
        self.buf.add(o, a, r - self.reward_shift, o2, done)
        self.steps += 1
        if self.buf.n >= self.learn_start and self.steps % self.train_every == 0:
            self.learn()

    def learn(self):
        c, m, g, a, r, c2, m2, g2, d = self.buf.sample(self.batch)
        q = self.q(c, g).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            qt_next = self.qt(c2, g2).masked_fill(m2 <= 0, NEG)
            if self.double:
                a2 = self.q(c2, g2).masked_fill(m2 <= 0, NEG).argmax(dim=1, keepdim=True)
                v_next = qt_next.gather(1, a2).squeeze(1)
            else:
                v_next = qt_next.max(dim=1).values
            y = r + self.gamma * (1.0 - d) * v_next
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        self.last_loss = float(loss.item())
        if self.updates % self.target_every == 0:
            self.qt.load_state_dict(self.q.state_dict())

    # ---------------------------------------------------------------- persist
    def state_dict(self):
        return {"q": self.q.state_dict(), "qt": self.qt.state_dict(), "opt": self.opt.state_dict(),
                "steps": self.steps, "updates": self.updates, "reward_shift": self.reward_shift}

    def load_state_dict(self, sd):
        self.q.load_state_dict(sd["q"])
        self.qt.load_state_dict(sd["qt"])
        if "opt" in sd:
            self.opt.load_state_dict(sd["opt"])
        self.steps, self.updates = sd.get("steps", 0), sd.get("updates", 0)
        self.reward_shift = sd.get("reward_shift", self.reward_shift)
