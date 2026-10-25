"""Agents for the GC-timing problem (2 actions: wait / GC now), with semi-Markov discounting.

TabularQ — lecture week 4 Q-learning on a discretised state:
    Q(s,a) <- Q(s,a) + alpha [ r + gamma^dt * max_a' Q(s',a') - Q(s,a) ]
TimingDQN — lecture week 6 DQN (MLP, replay buffer, target network, Double-DQN target) on the
5 raw features, same semi-Markov target.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

# bins for the tabular agent (on the env's normalised features)
FREE_BINS = 17                     # free blocks 0..16+ (obs[0] * 16)
IDLE_EDGES_MS = (0.25, 0.75, 1.5, 3, 6, 12, 24, 48, 96)   # -> 10 bins
U_EDGES = (0.5, 0.6, 0.7, 0.8)     # -> 5 bins


def tab_index(obs) -> int:
    free = min(int(round(obs[0] * 16.0)), FREE_BINS - 1)
    idle_ms = math.expm1(obs[1] * math.log1p(256.0))
    ib = sum(idle_ms >= e for e in IDLE_EDGES_MS)
    ub = sum(obs[3] >= e for e in U_EDGES)
    return (free * (len(IDLE_EDGES_MS) + 1) + ib) * (len(U_EDGES) + 1) + ub


N_TAB_STATES = FREE_BINS * (len(IDLE_EDGES_MS) + 1) * (len(U_EDGES) + 1)


class TabularQ:
    def __init__(self, gamma=0.995, alpha=0.1, eps_start=1.0, eps_end=0.02, eps_decay_steps=200_000, seed=0, **_):
        self.name = "tabular-q"
        self.Q = np.zeros((N_TAB_STATES, 2))
        self.N = np.zeros((N_TAB_STATES, 2), dtype=np.int64)
        self.gamma, self.alpha = gamma, alpha
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_steps
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.last_loss = float("nan")

    def epsilon(self):
        f = min(1.0, self.steps / max(self.eps_decay, 1))
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(2))
        q = self.Q[tab_index(obs)]
        return int(np.argmax(q)) if q[0] != q[1] else 0

    def observe(self, o, a, r, o2, done, dt_ms=1.0):
        s, s2 = tab_index(o), tab_index(o2)
        target = r + (0.0 if done else self.gamma ** dt_ms * self.Q[s2].max())
        td = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td
        self.N[s, a] += 1
        self.steps += 1
        self.last_loss = float(td * td)

    def state_dict(self):
        return {"Q": self.Q, "N": self.N, "steps": self.steps}

    def load_state_dict(self, sd):
        self.Q, self.N, self.steps = sd["Q"], sd["N"], sd["steps"]


class TimingDQN:
    def __init__(self, n_feat=5, gamma=0.995, lr=5e-4, hidden=64, batch=64, buffer=100_000, train_every=4,
                 target_every=500, learn_start=2000, eps_start=1.0, eps_end=0.02, eps_decay_steps=200_000,
                 double=True, seed=0, **_):
        torch.manual_seed(seed)
        self.name = "timing-dqn"
        mk = lambda: nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),  # noqa: E731
                                   nn.Linear(hidden, 2))
        self.q, self.qt = mk(), mk()
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.batch, self.train_every, self.target_every = gamma, batch, train_every, target_every
        self.learn_start, self.double = learn_start, double
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_steps
        self.rng = np.random.default_rng(seed + 3)
        self.cap, self.n, self.i = buffer, 0, 0
        self.o = np.zeros((buffer, n_feat), np.float32)
        self.o2 = np.zeros((buffer, n_feat), np.float32)
        self.a = np.zeros(buffer, np.int64)
        self.r = np.zeros(buffer, np.float32)
        self.g = np.zeros(buffer, np.float32)       # gamma ** dt per transition
        self.steps = self.updates = 0
        self.last_loss = float("nan")

    def epsilon(self):
        f = min(1.0, self.steps / max(self.eps_decay, 1))
        return self.eps_start + f * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(2))
        return int(self.q(torch.from_numpy(obs).unsqueeze(0)).argmax(1).item())

    def observe(self, o, a, r, o2, done, dt_ms=1.0):
        i = self.i
        self.o[i], self.a[i], self.r[i], self.o2[i] = o, a, r, o2
        self.g[i] = 0.0 if done else self.gamma ** dt_ms
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)
        self.steps += 1
        if self.n >= self.learn_start and self.steps % self.train_every == 0:
            self.learn()

    def learn(self):
        idx = self.rng.integers(0, self.n, size=self.batch)
        t = torch.from_numpy
        o, a, r, o2, g = t(self.o[idx]), t(self.a[idx]), t(self.r[idx]), t(self.o2[idx]), t(self.g[idx])
        q = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                a2 = self.q(o2).argmax(1, keepdim=True)
                v2 = self.qt(o2).gather(1, a2).squeeze(1)
            else:
                v2 = self.qt(o2).max(1).values
            y = r + g * v2
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        self.last_loss = float(loss.item())
        if self.updates % self.target_every == 0:
            self.qt.load_state_dict(self.q.state_dict())

    def state_dict(self):
        return {"q": self.q.state_dict(), "qt": self.qt.state_dict(), "opt": self.opt.state_dict(),
                "steps": self.steps, "updates": self.updates}

    def load_state_dict(self, sd):
        self.q.load_state_dict(sd["q"])
        self.qt.load_state_dict(sd["qt"])
        if "opt" in sd:
            self.opt.load_state_dict(sd["opt"])
        self.steps, self.updates = sd.get("steps", 0), sd.get("updates", 0)
