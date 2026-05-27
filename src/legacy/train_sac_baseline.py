#!/usr/bin/env python3
"""
SAC+MLP and SAC+LSTM baselines for MetaWorld PO experiments.

LSTM variant: actor uses LSTM for inference (step-by-step), but training uses
the same MLP critic. The LSTM actor is trained by unrolling episodes sequentially
rather than sampling from a replay buffer.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def make_env(task_name, seed, flash_every=20, flash_len=1, full_observable=False, po_mode="weak", memory_mode="none"):
    import metaworld
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ml1 = metaworld.ML1(task_name, seed=seed)
    env = ml1.train_classes[task_name]()
    env.set_task(ml1.train_tasks[0])
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._seed = seed
    from run_metaworld_mem import PartialObsWrapper
    env = PartialObsWrapper(
        env,
        po_mode=po_mode,
        memory_mode=memory_mode,
        flash_every=flash_every,
        flash_len=flash_len,
        mask_goal_after_reset=not full_observable,
        mask_goal_on_reset=False,
    )
    return env


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class LSTMActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden
        self.lstm_num_layers = lstm_layers
        self.lstm = nn.LSTM(obs_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.hidden = None

    def forward(self, obs):
        if obs.dim() == 1:
            x = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 2:
            x = obs.unsqueeze(1)
        else:
            x = obs
        if self.hidden is None:
            h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size, device=obs.device)
            c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size, device=obs.device)
            self.hidden = (h0, c0)
        out, self.hidden = self.lstm(x, self.hidden)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        return self.fc(out.squeeze(1))

    def reset_hidden(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        self.hidden = (h0, c0)

    def detach_hidden(self):
        if self.hidden is not None:
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs_buf[idx]),
            torch.tensor(self.act_buf[idx]),
            torch.tensor(self.rew_buf[idx]).unsqueeze(1),
            torch.tensor(self.next_obs_buf[idx]),
            torch.tensor(self.done_buf[idx]).unsqueeze(1),
        )


def _save_training_checkpoint(
    out_dir,
    actor,
    critic,
    critic_target,
    actor_optimizer,
    critic_optimizer,
    log_alpha,
    alpha_optimizer,
    global_step,
    episode_count,
    results,
):
    checkpoint = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "critic_target": critic_target.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "alpha_optimizer": alpha_optimizer.state_dict(),
        "global_step": global_step,
        "episode_count": episode_count,
        "results": results,
    }
    torch.save(checkpoint, out_dir / "checkpoint_latest.pt")
    torch.save(checkpoint, out_dir / f"checkpoint_step{global_step}.pt")
    torch.save(actor.state_dict(), out_dir / "actor.pt")


def _maybe_resume_training(
    args,
    out_dir,
    device,
    actor,
    critic,
    critic_target,
    actor_optimizer,
    critic_optimizer,
    log_alpha,
    alpha_optimizer,
):
    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.resume:
        candidate = out_dir / "checkpoint_latest.pt"
        if candidate.exists():
            resume_path = candidate

    if resume_path is None or not resume_path.exists():
        return log_alpha, 0, 0, []

    checkpoint = torch.load(resume_path, map_location=device)

    # Support both full training checkpoints and legacy actor-only checkpoints.
    if isinstance(checkpoint, dict) and "actor" in checkpoint:
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        critic_target.load_state_dict(checkpoint["critic_target"])
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        loaded_log_alpha = checkpoint.get("log_alpha", torch.zeros(1))
        log_alpha = loaded_log_alpha.to(device).detach().requires_grad_(True)
        alpha_optimizer = optim.Adam([log_alpha], lr=3e-4)
        alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        episode_count = int(checkpoint.get("episode_count", 0))
        results = list(checkpoint.get("results", []))
        print(f"[RESUME] Loaded full checkpoint: {resume_path} (global_step={global_step})")
        return log_alpha, global_step, episode_count, results

    actor.load_state_dict(checkpoint)
    print(f"[RESUME] Loaded legacy actor-only checkpoint: {resume_path} (training restarts from step 0)")
    return log_alpha, 0, 0, []


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) / f"{args.task}-{args.policy}-seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_obs_mode = str(getattr(args, "train_observation", "fo")).strip().lower()
    if train_obs_mode not in {"fo", "weak", "strong"}:
        raise ValueError(f"Unsupported train_observation: {train_obs_mode}")

    env = make_env(
        args.task,
        args.seed,
        flash_every=args.flash_every,
        flash_len=args.flash_len,
        full_observable=(train_obs_mode == "fo"),
        po_mode="weak" if train_obs_mode == "fo" else train_obs_mode,
        memory_mode=args.memory_mode,
    )
    obs_dim = env.reset()[0].shape[0] if isinstance(env.reset(), tuple) else env.reset().shape[0]
    act_dim = 4

    if args.policy == "mlp":
        actor = MLPActor(obs_dim, act_dim).to(device)
    else:
        actor = LSTMActor(obs_dim, act_dim).to(device)

    critic = MLPCritic(obs_dim, act_dim).to(device)
    critic_target = MLPCritic(obs_dim, act_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False

    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -act_dim

    log_alpha, global_step, episode_count, results = _maybe_resume_training(
        args,
        out_dir,
        device,
        actor,
        critic,
        critic_target,
        actor_optimizer,
        critic_optimizer,
        log_alpha,
        alpha_optimizer,
    )
    alpha_optimizer = optim.Adam([log_alpha], lr=3e-4)
    if args.resume_from or args.resume:
        resume_path = Path(args.resume_from) if args.resume_from else (out_dir / "checkpoint_latest.pt")
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device)
            if isinstance(checkpoint, dict) and "alpha_optimizer" in checkpoint:
                alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

    obs_raw = env.reset(seed=args.seed)
    obs = obs_raw[0] if isinstance(obs_raw, tuple) else obs_raw
    if args.policy == "lstm":
        actor.reset_hidden(batch_size=1, device=device)

    episode_reward = 0
    start_time = time.time()

    for global_step in range(global_step + 1, args.total_timesteps + 1):

        if global_step < args.learning_starts:
            action = np.random.uniform(-1, 1, size=act_dim).astype(np.float32)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                if args.policy == "lstm":
                    action = actor(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()
                else:
                    action = actor(obs_t).cpu().numpy()
            action = np.clip(action, -1, 1)

        next_obs_raw = env.step(action)
        if len(next_obs_raw) == 5:
            next_obs, reward, terminated, truncated, info = next_obs_raw
        else:
            next_obs, reward, done_flag, info = next_obs_raw
            terminated = done_flag
            truncated = False
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, float(done))
        obs = next_obs
        episode_reward += reward

        if done:
            results.append(episode_reward)
            episode_count += 1
            episode_reward = 0
            obs_raw = env.reset()
            obs = obs_raw[0] if isinstance(obs_raw, tuple) else obs_raw
            if args.policy == "lstm":
                actor.reset_hidden(batch_size=1, device=device)

        if global_step >= args.learning_starts and global_step % args.train_frequency == 0:
            batch = buffer.sample(args.batch_size)
            b_obs, b_act, b_rew, b_next_obs, b_done = [x.to(device) for x in batch]

            with torch.no_grad():
                if args.policy == "lstm":
                    actor.reset_hidden(batch_size=args.batch_size, device=device)
                    next_act = actor(b_next_obs)
                else:
                    next_act = actor(b_next_obs)
                next_q1, next_q2 = critic_target(b_next_obs, next_act)
                next_q = torch.min(next_q1, next_q2) - torch.exp(log_alpha) * next_act.pow(2).sum(-1, keepdim=True)
                target_q = b_rew + (1 - b_done) * args.gamma * next_q

            curr_q1, curr_q2 = critic(b_obs, b_act)
            critic_loss = F.mse_loss(curr_q1, target_q) + F.mse_loss(curr_q2, target_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if args.policy == "lstm":
                actor.reset_hidden(batch_size=args.batch_size, device=device)
                new_act = actor(b_obs)
            else:
                new_act = actor(b_obs)
            q1, q2 = critic(b_obs, new_act)
            q = torch.min(q1, q2)
            actor_loss = (torch.exp(log_alpha) * new_act.pow(2).sum(-1, keepdim=True) - q).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            alpha_loss = -(log_alpha * (new_act.pow(2).sum(-1, keepdim=True) + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            if global_step % 1000 == 0:
                for p, tp in zip(critic.parameters(), critic_target.parameters()):
                    tp.data.mul_(args.tau).add_((1 - args.tau) * p.data)

            if args.policy == "lstm":
                actor.reset_hidden(batch_size=1, device=device)

        if global_step % 10000 == 0:
            recent = results[-100:] if len(results) >= 100 else results
            mean_r = np.mean(recent) if recent else 0
            elapsed = time.time() - start_time
            print(f"[{global_step}/{args.total_timesteps}] episodes={episode_count} mean_reward(100)={mean_r:.2f} alpha={torch.exp(log_alpha).item():.4f} time={elapsed:.0f}s")

        if args.save_interval > 0 and global_step % args.save_interval == 0:
            _save_training_checkpoint(
                out_dir,
                actor,
                critic,
                critic_target,
                actor_optimizer,
                critic_optimizer,
                log_alpha,
                alpha_optimizer,
                global_step,
                episode_count,
                results,
            )

    _save_training_checkpoint(
        out_dir,
        actor,
        critic,
        critic_target,
        actor_optimizer,
        critic_optimizer,
        log_alpha,
        alpha_optimizer,
        global_step,
        episode_count,
        results,
    )
    torch.save(actor.state_dict(), out_dir / "actor.pt")
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(
            {
                "rewards": results,
                "total_steps": global_step,
                "episodes": episode_count,
                "save_interval": args.save_interval,
                "total_timesteps": args.total_timesteps,
            },
            f,
        )

    env.close()
    return out_dir


def evaluate(args, actor_path, full_observable, n_episodes=60, po_mode="weak", tag=None):
    import metaworld
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(
        args.task,
        args.seed,
        flash_every=args.flash_every,
        flash_len=args.flash_len,
        full_observable=full_observable,
        po_mode=po_mode,
        memory_mode=args.memory_mode,
    )
    obs_raw = env.reset(seed=args.seed)
    obs_dim = (obs_raw[0] if isinstance(obs_raw, tuple) else obs_raw).shape[0]
    act_dim = 4

    if args.policy == "mlp":
        actor = MLPActor(obs_dim, act_dim).to(device)
    else:
        actor = LSTMActor(obs_dim, act_dim).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    successes = 0
    total_rewards = []

    for ep in range(n_episodes):
        obs_raw = env.reset(seed=args.seed * 10000 + ep)
        obs = obs_raw[0] if isinstance(obs_raw, tuple) else obs_raw
        if args.policy == "lstm":
            actor.reset_hidden(batch_size=1, device=device)
        ep_reward = 0
        ep_success = 0.0
        done = False
        steps = 0

        while not done and steps < args.max_episode_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                if args.policy == "lstm":
                    action = actor(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()
                else:
                    action = actor(obs_t).cpu().numpy()
            action = np.clip(action, -1, 1)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                obs, reward, done_flag, info = step_result
                terminated = done_flag
                truncated = False
            ep_reward += reward
            ep_success = max(ep_success, float(info.get("success", 0.0) or 0.0))
            done = terminated or truncated
            steps += 1

        total_rewards.append(ep_reward)
        successes += ep_success

    env.close()
    result = {
        "task": args.task,
        "policy": args.policy,
        "seed": args.seed,
        "condition": tag or ("FO" if full_observable else po_mode.upper()),
        "po_mode": po_mode,
        "full_observable": bool(full_observable),
        "flash_every": args.flash_every,
        "flash_len": args.flash_len,
        "memory_mode": args.memory_mode,
        "episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(total_rewards)),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--policy", type=str, choices=["mlp", "lstm"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="artifacts/sac_baselines")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--actor-path", type=str, default=None)
    parser.add_argument("--flash-every", type=int, default=5)
    parser.add_argument("--flash-len", type=int, default=1)
    parser.add_argument("--po-mode", choices=["weak", "strong"], default="weak")
    parser.add_argument("--train-observation", choices=["fo", "weak", "strong"], default="fo")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--memory-mode", choices=["none", "goal_buffer"], default="none")
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    if args.eval_only:
        assert args.actor_path is not None
        eval_specs = [
            ("FO", True, "weak"),
            ("WeakPO", False, "weak"),
            ("StrongPO", False, "strong"),
        ]
        for tag, fo, po_mode in eval_specs:
            result = evaluate(args, args.actor_path, full_observable=fo, po_mode=po_mode, tag=tag)
            print(f"[{tag}] success_rate={result['success_rate']:.3f} mean_reward={result['mean_reward']:.2f}")
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            memory_suffix = "" if args.memory_mode == "none" else f"__{args.memory_mode}"
            eval_path = out_dir / f"{args.task}__{args.policy}__s{args.seed}{memory_suffix}__{tag}.eval.json"
            with open(eval_path, "w") as f:
                json.dump(result, f, indent=2)
    else:
        out_dir = train(args)
        actor_path = out_dir / "actor.pt"
        print(f"\n=== Evaluation ===")
        for tag, fo, po_mode in [("FO", True, "weak"), ("WeakPO", False, "weak"), ("StrongPO", False, "strong")]:
            result = evaluate(args, actor_path, full_observable=fo, po_mode=po_mode, tag=tag)
            print(f"[{tag}] success_rate={result['success_rate']:.3f} mean_reward={result['mean_reward']:.2f}")
