# Cart Pole

Now we discretize the initial starting state and learn a different action sequence for each bin.

### evolve an action sequence to solve cart pole
### discretize 4 continuous variables ranges.
### the four contiuous variables are binned into 3 buckets each (so 3**4 bins here)
### more/finer bins should lead to a more appropriate action sequence chosen at test time
```
uv run evo_cartpole_bin.py evolve --generations 50 --grid-bins 3
```

### demo the best action sequence found (for the closest initial condition)
```
uv run evo_cartpole_bin.py demo --sequence-file evo_cartpole_out/best_sequences_grid.npy --seed 123 --grid-bins 3
```