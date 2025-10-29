#!/usr/bin/env python3
"""
evo_cartpole.py — Evolutionary Algorithm with Grid of Initial Conditions

This version evolves the best action sequence for each discretized
initial condition in CartPole-v1, with verification of initial-state ranges
during both training and demo.

Usage examples:
    # Evolve across grid of initial conditions
    python evo_cartpole.py evolve --generations 100 --grid-bins 3

    # Demo adaptive playback on unseen random seed
    python evo_cartpole.py demo --sequence-file evo_cartpole_out/best_sequences_grid.npy --seed 17
"""

import argparse
import numpy as np
import json
import os
import time
from tqdm import tqdm
import gymnasium as gym


# ----------------------------
# Utilities
# ----------------------------
def seeded_env(base_seed: int = None, render_mode: str = None):
    if render_mode is None:
        env = gym.make("CartPole-v1")
    else:
        env = gym.make("CartPole-v1", render_mode=render_mode)
    return env


def evaluate_sequence_return_fitness(sequence: np.ndarray, init_state: np.ndarray) -> int:
    """Evaluate a sequence starting from a given initial state."""
    env = seeded_env()
    obs, _ = env.reset()
    env.unwrapped.state = init_state.copy()
    total_reward = 0
    for a in sequence:
        obs, reward, terminated, truncated, _ = env.step(int(a))
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    return int(total_reward)


def discretize_obs(obs: np.ndarray, bins_per_dim=3, low=None, high=None):
    """Discretize continuous 4D observation into integer bin indices (true CartPole init range)."""
    if low is None:
        low = np.array([-0.05, -0.05, -0.05, -0.05])
    if high is None:
        high = np.array([0.05, 0.05, 0.05, 0.05])
    bins = np.linspace(0, 1, bins_per_dim + 1)[1:-1]
    scaled = (obs - low) / (high - low)
    scaled = np.clip(scaled, 0, 1)
    indices = np.digitize(scaled, bins)
    return tuple(indices.tolist())


def center_of_bin(indices, bins_per_dim=3, low=None, high=None):
    """Compute representative continuous state for a given bin index tuple."""
    if low is None:
        low = np.array([-0.05, -0.05, -0.05, -0.05])
    if high is None:
        high = np.array([0.05, 0.05, 0.05, 0.05])
    edges = np.linspace(0, 1, bins_per_dim + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    frac = np.array([centers[i] for i in indices])
    return low + frac * (high - low)


# ----------------------------
# Evolutionary Island
# ----------------------------
class IslandOptimizer:
    def __init__(self, seq_len, pop_size, mutation_rate, elite_frac, init_state, rng):
        self.seq_len = seq_len
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elite_frac = elite_frac
        self.init_state = init_state
        self.rng = rng

        self.num_elite = max(1, int(np.ceil(self.pop_size * self.elite_frac)))
        self.population = self.rng.randint(0, 2, size=(self.pop_size, self.seq_len), dtype=np.uint8)
        self.fitness = np.zeros(self.pop_size, dtype=np.int32)
        self.best_seq = None
        self.best_fitness = -1

    def evaluate(self):
        for i in range(self.pop_size):
            seq = self.population[i]
            fit = evaluate_sequence_return_fitness(seq, self.init_state)
            self.fitness[i] = fit
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_seq = seq.copy()

    def select_elites(self):
        elite_idx = np.argsort(self.fitness)[-self.num_elite:][::-1]
        return self.population[elite_idx].copy()

    def make_offspring(self, target_count):
        offspring = np.zeros((target_count, self.seq_len), dtype=np.uint8)
        for j in range(target_count):
            def tournament():
                choices = self.rng.choice(self.pop_size, size=3, replace=False)
                return choices[np.argmax(self.fitness[choices])]

            p1 = self.population[tournament()]
            p2 = self.population[tournament()]
            cp = self.rng.randint(1, self.seq_len)
            child = np.concatenate([p1[:cp], p2[cp:]])
            mask = self.rng.random(self.seq_len) < self.mutation_rate
            if mask.any():
                flip = self.rng.randint(0, 2, size=self.seq_len, dtype=np.uint8)
                child = np.where(mask, flip, child)
            offspring[j] = child
        return offspring

    def next_generation_from_elites(self, elites):
        next_pop = np.zeros_like(self.population)
        n_elites = min(len(elites), self.pop_size)
        next_pop[:n_elites] = elites
        remaining = self.pop_size - n_elites
        if remaining > 0:
            offspring = self.make_offspring(remaining)
            next_pop[n_elites:] = offspring
        self.population = next_pop
        self.fitness.fill(0)
        self.best_fitness = -1


# ----------------------------
# Evolution Runners
# ----------------------------
def run_evolution_for_state(args, init_state):
    rng = np.random.RandomState(args.seed)
    isl = IslandOptimizer(args.seq_len, args.pop_size, args.mutation_rate,
                          args.elite_frac, init_state, rng)
    for gen in tqdm(range(args.generations), desc="Generations"):
        isl.evaluate()
        elites = isl.select_elites()
        isl.next_generation_from_elites(elites)
    isl.evaluate()
    return isl.best_seq, isl.best_fitness


def run_evolution_grid(args):
    """Run evolution for each discretized bin with verification printout."""
    grid_bins = args.grid_bins
    best_sequences = {}
    total_cells = grid_bins ** 4
    cell = 0

    for i in range(grid_bins):
        for j in range(grid_bins):
            for k in range(grid_bins):
                for l in range(grid_bins):
                    cell += 1
                    init_state = center_of_bin((i, j, k, l), bins_per_dim=grid_bins)

                    # ✅ Verification print
                    print(
                        f"\n[{cell}/{total_cells}] Training for bin ({i},{j},{k},{l}) "
                        f"with initial state center = {np.round(init_state, 5)}"
                    )

                    best_seq, best_fit = run_evolution_for_state(args, init_state)
                    best_sequences[(i, j, k, l)] = {
                        "sequence": best_seq.tolist(),
                        "fitness": int(best_fit),
                        "init_state": init_state.tolist(),
                    }

    os.makedirs(args.out_dir, exist_ok=True)
    grid_file = os.path.join(args.out_dir, "best_sequences_grid.npy")
    np.save(grid_file, best_sequences)
    print(f"\n✅ Saved grid of best sequences to {grid_file}")
    return grid_file


# ----------------------------
# Demo runner
# ----------------------------
def demo_sequence(sequence_file: str, render_mode="human", seed=42, pause=0.02, grid_bins=3):
    """Demo: verify actual random start, check range, and use best bin sequence."""
    env = seeded_env(render_mode=render_mode)
    obs, _ = env.reset(seed=seed)

    print("\n===== DEMO START =====")
    print(f"Random seed: {seed}")
    print(f"Observed initial state: {np.round(obs, 6)}")

    # ✅ Range check
    within_range = np.all((obs >= -0.05) & (obs <= 0.05))
    if within_range:
        print("✅ Initial state lies within expected CartPole range (-0.05, 0.05).")
    else:
        print("⚠️  Initial state exceeds expected range (-0.05, 0.05):")
        for idx, val in enumerate(obs):
            if val < -0.05 or val > 0.05:
                print(f"   dim {idx}: {val:.6f}")

    # ✅ Load the grid and pick bin
    grid = np.load(sequence_file, allow_pickle=True).item()
    idx = discretize_obs(obs, bins_per_dim=grid_bins)
    print(f"Discretized bin index = {idx}")

    if idx not in grid:
        all_keys = np.array(list(grid.keys()))
        dists = np.linalg.norm(all_keys - np.array(idx), axis=1)
        idx = tuple(all_keys[np.argmin(dists)])
        print(f"Closest existing bin used: {idx}")

    seq = np.array(grid[idx]["sequence"], dtype=np.uint8)
    init_state = np.array(grid[idx]["init_state"])
    print(f"Using bin {idx} with representative center = {np.round(init_state, 5)}")
    print("======================\n")

    env.unwrapped.state = obs.copy()
    total_reward = 0
    for a in seq:
        obs, reward, terminated, truncated, _ = env.step(int(a))
        total_reward += reward
        try:
            env.render()
        except Exception:
            pass
        time.sleep(pause)
        if terminated or truncated:
            break
    env.close()
    print(f"Demo finished. Steps survived: {total_reward}/{len(seq)}")


# ----------------------------
# CLI
# ----------------------------
def make_parser():
    p = argparse.ArgumentParser(description="Evolve CartPole sequences with grid of initial conditions")
    sub = p.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("evolve")
    t.add_argument("--generations", type=int, default=200)
    t.add_argument("--pop-size", type=int, default=200)
    t.add_argument("--seq-len", type=int, default=200)
    t.add_argument("--mutation-rate", type=float, default=0.02)
    t.add_argument("--elite-frac", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--grid-bins", type=int, default=1)
    t.add_argument("--out-dir", type=str, default="evo_cartpole_out")

    d = sub.add_parser("demo")
    d.add_argument("--sequence-file", type=str, required=True)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--pause", type=float, default=0.02)
    d.add_argument("--render-mode", type=str, default="human")
    d.add_argument("--grid-bins", type=int, default=3)

    return p


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.mode == "evolve":
        if args.grid_bins > 1:
            run_evolution_grid(args)
        else:
            env = seeded_env()
            obs, _ = env.reset(seed=args.seed)
            best_seq, best_fit = run_evolution_for_state(args, obs)
            os.makedirs(args.out_dir, exist_ok=True)
            out_file = os.path.join(args.out_dir, "best_sequence.npy")
            np.save(out_file, best_seq)
            print(f"✅ Saved best sequence (fitness={best_fit}) to {out_file}")

    elif args.mode == "demo":
        demo_sequence(args.sequence_file, render_mode=args.render_mode,
                      seed=args.seed, pause=args.pause, grid_bins=args.grid_bins)


if __name__ == "__main__":
    main()
