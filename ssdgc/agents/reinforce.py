"""REINFORCE with a learned baseline (lecture week 7) for victim selection — written from scratch.

Policy  pi_theta(i | s) = softmax_i( f_theta([cand_i, glob]) )  over the real candidates
        (the same per-candidate scorer as the DQN, but its output is a logit, not a Q value).
Baseline V_phi(s) = MLP([glob, masked mean of candidates, masked max of candidates]).

One update per episode (a truncated GC run of `episode_len` decisions):
    R_t   = sum_{k>=t} gamma^(k-t) r_k                      (reward-to-go, computed backwards)
    A_t   = R_t - V_phi(s_t)                                (normalised over the episode)
    L_pi  = -mean_t[ log pi_theta(a_t | s_t) * A_t ] - beta * mean_t[ H(pi_theta(. | s_t)) ]
    L_V   = mean_t[ (R_t - V_phi(s_t))^2 ]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dqn import NEG, Scorer


class ValueNet(nn.Module):
    def __init__(self, F: int, G: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(G + 2 * F, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, cand, mask, glob):
        m = mask.unsqueeze(-1)
        mean = (cand * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = cand.masked_fill(m <= 0, -1.0).max(1).values
        return self.net(torch.cat([glob, mean, mx], dim=-1)).squeeze(-1)


class ReinforceAgent:
    def __init__(self, k: int, F: int, G: int, gamma: float = 0.99, lr: float = 3e-4, lr_v: float = 1e-3,
                 hidden: int = 64, entropy: float = 0.01, reward_shift: float = 0.0, seed: int = 0, **_):
        torch.manual_seed(seed)
        self.name = "reinforce"
        self.k, self.gamma, self.entropy = k, gamma, entropy
        self.pi = Scorer(F, G, hidden)
        self.v = ValueNet(F, G, max(hidden, 64))
        self.opt = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=lr_v)
        self.reward_shift = reward_shift
        self.rng = np.random.default_rng(seed + 29)
        self.steps = 0
        self.updates = 0
        self.last_loss = float("nan")
        self._ep = []

    @torch.no_grad()
    def probs(self, obs) -> np.ndarray:
        c = torch.from_numpy(obs["cand"]).unsqueeze(0)
        g = torch.from_numpy(obs["glob"]).unsqueeze(0)
        logits = self.pi(c, g)[0]
        logits = logits.masked_fill(torch.from_numpy(obs["mask"]) <= 0, NEG)
        return torch.softmax(logits, -1).numpy()

    def act(self, obs, greedy: bool = False) -> int:
        p = self.probs(obs)
        if greedy:
            return int(np.argmax(p))
        p = p / p.sum()
        return int(self.rng.choice(len(p), p=p))

    def observe(self, o, a, r, o2, done):
        self._ep.append((o["cand"], o["mask"], o["glob"], a, r - self.reward_shift))
        self.steps += 1

    def end_episode(self, last_obs=None):
        """One policy-gradient step on the finished episode.  If the episode was cut (truncated, not
        terminated), the return after the cut is filled in with the value net: G_T = V(s_T)."""
        if not self._ep:
            return
        cand = torch.from_numpy(np.stack([e[0] for e in self._ep]))
        mask = torch.from_numpy(np.stack([e[1] for e in self._ep]))
        glob = torch.from_numpy(np.stack([e[2] for e in self._ep]))
        act = torch.tensor([e[3] for e in self._ep], dtype=torch.int64)
        rew = np.array([e[4] for e in self._ep], dtype=np.float64)
        ret = np.zeros_like(rew)
        run = 0.0
        if last_obs is not None:
            with torch.no_grad():
                run = float(self.v(torch.from_numpy(last_obs["cand"]).unsqueeze(0),
                                   torch.from_numpy(last_obs["mask"]).unsqueeze(0),
                                   torch.from_numpy(last_obs["glob"]).unsqueeze(0))[0])
        for t in range(len(rew) - 1, -1, -1):
            run = rew[t] + self.gamma * run
            ret[t] = run
        R = torch.from_numpy(ret.astype(np.float32))
        v = self.v(cand, mask, glob)
        adv = (R - v.detach())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logits = self.pi(cand, glob).masked_fill(mask <= 0, NEG)
        logp_all = torch.log_softmax(logits, -1)
        logp = logp_all.gather(1, act.unsqueeze(1)).squeeze(1)
        p_all = logp_all.exp()
        ent = -(p_all * logp_all.masked_fill(mask <= 0, 0.0)).sum(-1)
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
        self.last_vloss = float(loss_v.item())
        self.last_entropy = float(ent.mean().item())
        self._ep = []

    def state_dict(self):
        return {"pi": self.pi.state_dict(), "v": self.v.state_dict(), "opt": self.opt.state_dict(),
                "opt_v": self.opt_v.state_dict(), "steps": self.steps, "updates": self.updates,
                "reward_shift": self.reward_shift}

    def load_state_dict(self, sd):
        self.pi.load_state_dict(sd["pi"])
        self.v.load_state_dict(sd["v"])
        if "opt" in sd:
            self.opt.load_state_dict(sd["opt"])
            self.opt_v.load_state_dict(sd["opt_v"])
        self.steps, self.updates = sd.get("steps", 0), sd.get("updates", 0)
        self.reward_shift = sd.get("reward_shift", self.reward_shift)
