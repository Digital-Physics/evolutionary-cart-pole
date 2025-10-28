#!/usr/bin/env python3
"""
evo_cartpole.py

Evolutionary Algorithm to evolve Left/Right action sequences (0/1) for CartPole-v1.

Features:
- Fixed-length action sequences (default length 200).
- Fitness = total reward (steps survived).
- Deterministic episode starts using a base_seed so evaluations are repeatable.
- Multiple "island" populations early; islands are merged gradually until 1 population remains.
- No masking of "non-coding" suffixes before mutation (simple implementation).
- CLI with evolve/demo modes.

Usage examples:
    python evo_cartpole.py evolve --generations 200 --pop-size 200 --seq-len 200
    python evo_cartpole.py demo --sequence-file best_sequence.npy --seed 42
"""

import argparse
import numpy as np
import json
import os
import time
from tqdm import tqdm
import gymnasium as gym
from typing import List, Tuple

# ----------------------------
# Utilities
# ----------------------------
def seeded_env(base_seed: int, render_mode: str = None):
    """Make a new CartPole env and return it. We do NOT set the seed globally here;
    reset() will receive the seed per-evaluation to enforce determinism per episode."""
    if render_mode is None:
        env = gym.make("CartPole-v1")
    else:
        env = gym.make("CartPole-v1", render_mode=render_mode)
    return env

def evaluate_sequence_return_fitness(sequence: np.ndarray, base_seed: int) -> int:
    """Evaluate one sequence deterministically using base_seed.
    Returns the total reward (int), i.e., number of steps survived (<= len(sequence)).
    Uses a fresh env for safety (gym is lightweight to create)."""
    env = seeded_env(base_seed)
    # reset with the same seed for deterministic start
    obs, info = env.reset(seed=base_seed)
    total_reward = 0
    for idx, action in enumerate(sequence):
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    return int(total_reward)

# ----------------------------
# Evolution helpers
# ----------------------------
class IslandOptimizer:
    """An island (subpopulation) for evolving sequences."""
    def __init__(self,
                 seq_len: int,
                 pop_size: int,
                 mutation_rate: float,
                 elite_frac: float,
                 base_seed: int,
                 rng: np.random.RandomState):
        self.seq_len = seq_len
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elite_frac = elite_frac
        self.base_seed = base_seed
        self.rng = rng

        self.num_elite = max(1, int(np.ceil(self.pop_size * self.elite_frac)))

        # population: array shape (pop_size, seq_len), dtype uint8 (0 or 1)
        self.population = self.rng.randint(0, 2, size=(self.pop_size, self.seq_len), dtype=np.uint8)
        self.fitness = np.zeros(self.pop_size, dtype=np.int32)

        # bookkeeping
        self.best_idx = 0
        self.best_fitness = -1
        self.best_seq = None

    def evaluate(self):
        """Evaluate full population. Uses the same base_seed for deterministic starts.
        Because we want the same initial state for each sequence evaluation within the run,
        we do env.reset(seed=self.base_seed) inside evaluate_sequence_return_fitness."""
        for i in range(self.pop_size):
            seq = self.population[i]
            # deterministic evaluation using same seed
            fit = evaluate_sequence_return_fitness(seq, base_seed=self.base_seed)
            self.fitness[i] = fit
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_seq = seq.copy()
                self.best_idx = i

    def select_elites(self) -> np.ndarray:
        """Return elite sequences (copy)."""
        elite_idx = np.argsort(self.fitness)[-self.num_elite:][::-1]
        return self.population[elite_idx].copy()

    def make_offspring(self, target_count: int) -> np.ndarray:
        """Produce target_count offspring via tournament or roulette + crossover + mutation.
        We'll use tournament selection (k=3) for simplicity and single-point crossover."""
        offspring = np.zeros((target_count, self.seq_len), dtype=np.uint8)
        for j in range(target_count):
            # tournament selection
            def tournament():
                choices = self.rng.choice(self.pop_size, size=3, replace=False)
                return choices[np.argmax(self.fitness[choices])]

            p1 = self.population[tournament()]
            p2 = self.population[tournament()]
            # single point crossover (avoid 0 or seq_len)
            if self.seq_len > 1:
                cp = self.rng.randint(1, self.seq_len)
                child = np.concatenate([p1[:cp], p2[cp:]])
            else:
                child = p1.copy()
            # mutate
            mask = self.rng.random(self.seq_len) < self.mutation_rate
            if mask.any():
                flip = self.rng.randint(0, 2, size=self.seq_len, dtype=np.uint8)
                child = np.where(mask, flip, child)
            offspring[j] = child
        return offspring

    def next_generation_from_elites(self, elites: np.ndarray):
        """Form next generation by keeping elites and filling with offspring."""
        next_pop = np.zeros_like(self.population)
        # place elites first (may be fewer than self.num_elite if elites has small shape)
        n_elites = min(len(elites), self.pop_size)
        next_pop[:n_elites] = elites[:n_elites]
        # fill remainder
        remaining = self.pop_size - n_elites
        if remaining > 0:
            offspring = self.make_offspring(remaining)
            next_pop[n_elites:] = offspring
        self.population = next_pop
        # reset fitness bookkeeping
        self.fitness.fill(0)
        self.best_fitness = -1
        self.best_seq = None
        self.best_idx = 0

# ----------------------------
# Orchestration (islands + merging)
# ----------------------------
def run_evolution(args):
    """
    Evolve with islands that merge over time.
    Strategy:
      - Start with `start_islands` islands, each with pop_size = args.pop_size // start_islands (plus remainder)
      - Every `merge_every` generations, reduce the number of islands by half (rounded up),
        by collecting the top sequences across islands and re-splitting new islands with
        elites + mutated children. Continue until 1 island remains.
    """
    rng = np.random.RandomState(args.seed)

    # compute initial islands sizes
    start_islands = max(1, args.start_islands)
    islands = []
    base_pop = args.pop_size // start_islands
    remainder = args.pop_size % start_islands
    pop_sizes = [base_pop + (1 if i < remainder else 0) for i in range(start_islands)]

    # instantiate islands
    for ps in pop_sizes:
        islands.append(
            IslandOptimizer(seq_len=args.seq_len,
                            pop_size=ps,
                            mutation_rate=args.mutation_rate,
                            elite_frac=args.elite_frac,
                            base_seed=args.base_seed,
                            rng=np.random.RandomState(rng.randint(0, 2**31)))
        )

    generation = 0
    global_best = {'fitness': -1, 'seq': None, 'gen': -1}
    all_time_best_list = []  # keep top N all-time

    print(f"EVOLVE: seq_len={args.seq_len}, pop_total={args.pop_size}, start_islands={len(islands)}")
    print(f"Generations: {args.generations}, merge_every={args.merge_every}, seed={args.seed}")
    print("Starting evolution...")

    pbar = tqdm(total=args.generations, desc="Generations")
    while generation < args.generations:
        # Evaluate all islands
        for isl in islands:
            isl.evaluate()

        # Collect stats and update global best
        for idx, isl in enumerate(islands):
            if isl.best_fitness > global_best['fitness']:
                global_best['fitness'] = int(isl.best_fitness)
                global_best['seq'] = isl.best_seq.copy()
                global_best['gen'] = generation

        # Save snapshot info
        gen_snapshot = {
            'generation': generation,
            'num_islands': len(islands),
            'island_sizes': [isl.pop_size for isl in islands],
            'global_best_fitness': int(global_best['fitness']),
        }
        if args.verbose:
            pbar.set_description(f"Gen {generation} | islands {len(islands)} | best {global_best['fitness']}")

        # Merge logic: every merge_every generations, reduce islands by roughly half until 1
        will_merge = ((generation + 1) % args.merge_every == 0) and (len(islands) > 1)
        if will_merge:
            # collect top K sequences from each island
            top_k = args.top_k_from_island
            candidates = []
            for isl in islands:
                idxs = np.argsort(isl.fitness)[-min(top_k, isl.pop_size):][::-1]
                for i in idxs:
                    candidates.append((isl.population[i].copy(), int(isl.fitness[i])))
            # sort candidates and pick top args.keep_top_total
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            keep = candidates[:args.keep_top_total]
            # create a merged pool of sequences
            kept_seqs = np.array([s for s, f in keep], dtype=np.uint8)
            kept_fits = [f for s, f in keep]
            if args.verbose:
                print(f"\nMerging islands at gen {generation+1}: kept {len(kept_seqs)} sequences (top fits: {kept_fits})")

            # new island count is ceil(old / 2)
            new_island_count = max(1, (len(islands) + 1) // 2)
            # split pop sizes
            new_base = args.pop_size // new_island_count
            new_rem = args.pop_size % new_island_count
            new_pop_sizes = [new_base + (1 if i < new_rem else 0) for i in range(new_island_count)]

            # Re-seed RNGs for reproducibility
            new_rngs = [np.random.RandomState(rng.randint(0, 2**31)) for _ in range(new_island_count)]
            new_islands = []
            # For each new island, form population by filling with kept seqs (rotated) + offspring
            for i_isl, ps in enumerate(new_pop_sizes):
                isl_rng = new_rngs[i_isl]
                new_isl = IslandOptimizer(seq_len=args.seq_len,
                                          pop_size=ps,
                                          mutation_rate=args.mutation_rate,
                                          elite_frac=args.elite_frac,
                                          base_seed=args.base_seed,
                                          rng=isl_rng)
                # Fill elites by sampling from kept_seqs (or random if not enough)
                n_keep_here = min(len(kept_seqs), max(1, ps // 5))
                if n_keep_here > 0 and len(kept_seqs) > 0:
                    picks = isl_rng.choice(len(kept_seqs), size=n_keep_here, replace=True)
                    new_isl.population[:n_keep_here] = kept_seqs[picks]
                # Fill remainder randomly (will be overwritten by offspring step below if desired)
                if n_keep_here < ps:
                    new_isl.population[n_keep_here:] = isl_rng.randint(0, 2, size=(ps - n_keep_here, args.seq_len), dtype=np.uint8)
                new_islands.append(new_isl)

            # replace islands
            islands = new_islands

        else:
            # Normal evolution step within each island: select elites & produce next generation
            for isl in islands:
                elites = isl.select_elites()
                isl.next_generation_from_elites(elites)

        generation += 1
        pbar.update(1)

    pbar.close()

    # Final evaluation (ensure last generation has fitness evaluated)
    for isl in islands:
        isl.evaluate()
        if isl.best_fitness > global_best['fitness']:
            global_best['fitness'] = int(isl.best_fitness)
            global_best['seq'] = isl.best_seq.copy()
            global_best['gen'] = generation

    # Save final results
    os.makedirs(args.out_dir, exist_ok=True)
    best_seq_file = os.path.join(args.out_dir, "best_sequence.npy")
    np.save(best_seq_file, global_best['seq'])
    summary = {
        'best_fitness': int(global_best['fitness']),
        'best_gen': int(global_best['gen']),
        'seq_len': int(args.seq_len),
        'population': int(args.pop_size),
        'start_islands': int(args.start_islands),
        'final_islands': int(len(islands))
    }
    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nEvolution complete.")
    print(f"Best fitness: {global_best['fitness']} at generation {global_best['gen']}")
    print(f"Saved best sequence to: {best_seq_file}")
    return best_seq_file

# ----------------------------
# Demo runner
# ----------------------------
def demo_sequence(sequence_file: str, render_mode: str = "human", base_seed: int = 42, pause: float = 0.02):
    if not os.path.exists(sequence_file):
        raise FileNotFoundError(sequence_file)
    seq = np.load(sequence_file).astype(np.uint8)
    env = seeded_env(base_seed=base_seed, render_mode=render_mode)
    obs, info = env.reset(seed=base_seed)
    total_reward = 0
    step = 0
    for a in seq:
        obs, reward, terminated, truncated, info = env.step(int(a))
        total_reward += reward
        step += 1
        # If environment supports render() call (render_mode="human" created it) then call it.
        # Some Gym/Gymnasium backends auto-render at step if render_mode='human'.
        try:
            env.render()
        except Exception:
            pass
        time.sleep(pause)
        if terminated or truncated:
            break
    print(f"Demo finished. Steps survived (reward): {total_reward}/{len(seq)}")
    env.close()

# ----------------------------
# CLI
# ----------------------------
def make_parser():
    p = argparse.ArgumentParser(description="Evolve CartPole action sequences (0 = Left, 1 = Right)")
    sub = p.add_subparsers(dest="mode", required=True)

    # Evolve
    t = sub.add_parser("evolve", help="Run evolutionary algorithm")
    t.add_argument("--generations", type=int, default=200, help="Total generations")
    t.add_argument("--pop-size", type=int, default=200, help="Total population across islands")
    t.add_argument("--seq-len", type=int, default=200, help="Length of action sequences (max steps per episode)")
    t.add_argument("--mutation-rate", type=float, default=0.02, help="Per-gene mutation probability")
    t.add_argument("--elite-frac", type=float, default=0.1, help="Fraction of elites to keep per island")
    t.add_argument("--start-islands", type=int, default=8, help="Number of islands to start with")
    t.add_argument("--merge-every", type=int, default=20, help="Merge islands every N generations")
    t.add_argument("--top-k-from-island", type=int, default=5, help="How many top sequences to take from each island during merge")
    t.add_argument("--keep-top-total", type=int, default=30, help="How many top sequences to keep globally on merge")
    t.add_argument("--seed", type=int, default=42, help="Random seed (controls rng)")
    t.add_argument("--base-seed", type=int, default=42, help="Base seed used for env.reset -> deterministic starts")
    t.add_argument("--out-dir", type=str, default="evo_cartpole_out", help="Output directory")
    t.add_argument("--verbose", action="store_true")
    t.set_defaults(func=None)

    # Demo
    d = sub.add_parser("demo", help="Run demo of a saved sequence")
    d.add_argument("--sequence-file", type=str, default="best_sequence.npy", help="Numpy file containing best sequence")
    d.add_argument("--seed", type=int, default=42, help="Base seed for deterministic start")
    d.add_argument("--pause", type=float, default=0.02, help="Seconds between steps for visualization")
    d.add_argument("--render-mode", type=str, default="human", help="Render mode to pass to environment (human)") 

    return p

def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.mode == "evolve":
        # wire a few args to evolve function
        # rename attributes to match function signature
        evolve_args = argparse.Namespace(
            generations=args.generations,
            pop_size=args.pop_size,
            seq_len=args.seq_len,
            mutation_rate=args.mutation_rate,
            elite_frac=args.elite_frac,
            start_islands=args.start_islands,
            merge_every=args.merge_every,
            top_k_from_island=args.top_k_from_island,
            keep_top_total=args.keep_top_total,
            seed=args.seed,
            base_seed=args.base_seed,
            out_dir=args.out_dir,
            verbose=args.verbose
        )
        best_file = run_evolution(evolve_args)
        print("Done evolution. To demo, run:\n  python evo_cartpole.py demo --sequence-file", best_file)

    elif args.mode == "demo":
        demo_sequence(args.sequence_file, render_mode=args.render_mode, base_seed=args.seed, pause=args.pause)

if __name__ == "__main__":
    main()
