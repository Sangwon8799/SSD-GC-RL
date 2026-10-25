"""Agents for the rule-choosing formulation (a handful of numbers in, one of K rules out) — written from scratch.

RuleDQN       — lecture week 6: MLP Q(s, .) over K actions, replay buffer, target network, Double-DQN target
RuleReinforce — lecture week 7: softmax policy over K actions + value baseline, reward-to-go, truncation bootstrap
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(n_in, n_out, hidden=64):
    if hidden <= 0:
        return nn.Linear(n_in, n_out)
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_out))


class RuleDQN:
    def __init__(self, n_feat, n_act, gamma=0.9, lr=1e-3, hidden=64, batch=64, buffer=50_000, train_every=1,
                 target_every=200, learn_start=200, eps_start=1.0, eps_end=0.02, eps_decay_steps=2000,
                 double=True, reward_shift=0.0, seed=0, **_):
        torch.manual_seed(seed)
        self.name = "rule-dqn"
        self.q, self.qt = mlp(n_feat, n_act, hidden), mlp(n_feat, n_act, hidden)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.n_act = n_act
        self.gamma, self.batch, self.train_every, self.target_every = gamma, batch, train_every, target_every
        self.learn_start, self.double, self.reward_shift = learn_start, double, reward_shift
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_steps
        self.rng = np.random.default_rng(seed + 5)
        self.cap, self.n, self.i = buffer, 0, 0
        self.o = np.zeros((buffer, n_feat), np.float32)
        self.o2 = np.zeros((buffer, n_feat), np.float32)
        self.a = np.zeros(buffer, np.int64)
        self.r = np.zeros(buffer, np.float32)
        self.steps = self.updates = 0
        self.last_loss = float("nan")

    def epsilon(self):
        f = min(1.0, self.steps / max(self.eps_decay, 1))
        return self.eps_start + f * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_act))
        return int(self.q(torch.from_numpy(obs).unsqueeze(0)).argmax(1).item())

    def observe(self, o, a, r, o2, done, truncated=False):
        i = self.i
        self.o[i], self.a[i], self.r[i], self.o2[i] = o, a, r - self.reward_shift, o2
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)
        self.steps += 1
        if self.n >= self.learn_start and self.steps % self.train_every == 0:
            self.learn()

    def learn(self):
        idx = self.rng.integers(0, self.n, size=self.batch)
        t = torch.from_numpy
        o, a, r, o2 = t(self.o[idx]), t(self.a[idx]), t(self.r[idx]), t(self.o2[idx])
        q = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                a2 = self.q(o2).argmax(1, keepdim=True)
                v2 = self.qt(o2).gather(1, a2).squeeze(1)
            else:
                v2 = self.qt(o2).max(1).values
            y = r + self.gamma * v2
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
        return {"q": self.q.state_dict(), "qt": self.qt.state_dict(), "steps": self.steps,
                "updates": self.updates, "reward_shift": self.reward_shift}

    def load_state_dict(self, sd):
        self.q.load_state_dict(sd["q"])
        self.qt.load_state_dict(sd["qt"])
        self.steps, self.updates = sd.get("steps", 0), sd.get("updates", 0)
        self.reward_shift = sd.get("reward_shift", self.reward_shift)


class RuleReinforce:
    def __init__(self, n_feat, n_act, gamma=0.9, lr=1e-3, lr_v=1e-3, hidden=64, entropy=0.01, reward_shift=0.0,
                 seed=0, **_):
        torch.manual_seed(seed)
        self.name = "rule-reinforce"
        self.pi, self.v = mlp(n_feat, n_act, hidden), mlp(n_feat, 1, max(hidden, 64))
        self.opt = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=lr_v)
        self.gamma, self.entropy, self.reward_shift = gamma, entropy, reward_shift
        self.rng = np.random.default_rng(seed + 29)
        self.steps = self.updates = 0
        self.last_loss = float("nan")
        self._ep = []

    @torch.no_grad()
    def probs(self, obs):
        return torch.softmax(self.pi(torch.from_numpy(obs).unsqueeze(0))[0], -1).numpy()

    def act(self, obs, greedy=False):
        p = self.probs(obs)
        if greedy:
            return int(np.argmax(p))
        return int(self.rng.choice(len(p), p=p / p.sum()))

    def observe(self, o, a, r, o2, done, truncated=False):
        self._ep.append((o, a, r - self.reward_shift))
        self.steps += 1

    def end_episode(self, last_obs=None):
        if not self._ep:
            return
        obs = torch.from_numpy(np.stack([e[0] for e in self._ep]))
        act = torch.tensor([e[1] for e in self._ep], dtype=torch.int64)
        rew = np.array([e[2] for e in self._ep], dtype=np.float64)
        ret = np.zeros_like(rew)
        run = 0.0
        if last_obs is not None:
            with torch.no_grad():
                run = float(self.v(torch.from_numpy(last_obs).unsqueeze(0))[0, 0])
        for t in range(len(rew) - 1, -1, -1):
            run = rew[t] + self.gamma * run
            ret[t] = run
        R = torch.from_numpy(ret.astype(np.float32))
        v = self.v(obs).squeeze(-1)
        adv = R - v.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp_all = torch.log_softmax(self.pi(obs), -1)
        logp = logp_all.gather(1, act.unsqueeze(1)).squeeze(1)
        ent = -(logp_all.exp() * logp_all).sum(-1)
        loss_pi = -(logp * adv).mean() - self.entropy * ent.mean()
        self.opt.zero_grad()
        loss_pi.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), 5.0)
        self.opt.step()
        loss_v = ((R - v) ** 2).mean()
        self.opt_v.zero_grad()
        loss_v.backward()
        nn.utils.clip_grad_norm_(self.v.parameters(), 5.0)
        self.opt_v.step()
        self.updates += 1
        self.last_loss = float(loss_pi.item())
        self._ep = []

    def state_dict(self):
        return {"pi": self.pi.state_dict(), "v": self.v.state_dict(), "steps": self.steps, "updates": self.updates,
                "reward_shift": self.reward_shift}

    def load_state_dict(self, sd):
        self.pi.load_state_dict(sd["pi"])
        self.v.load_state_dict(sd["v"])
        self.steps, self.updates = sd.get("steps", 0), sd.get("updates", 0)
        self.reward_shift = sd.get("reward_shift", self.reward_shift)
