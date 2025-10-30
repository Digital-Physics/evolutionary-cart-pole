# Part 1: Evolutionary Cart Pole

This repository uses Evolutionary Algorithms to evolve a Left/Right action sequence solution. 

In part 1, we assume the pole is initialized the same, so we use the same random seed.
This deterministic environment should help learning. We can relax this assumption in part 2.  

We assume an episode can last at most 200 steps, and you get 1 reward per step. We generate sequences (a list) of length 200 "L"s and "R"s to start. This is an upper bound on the number of actions we'll need to take in an episode. At the end of the first generation, the fitness/total steps survived/total_reward will be used to determine who moves on and is mutated for the next generation.  

At mutation time, when some Ls get flipped to Rs (or vice-versa), we could either mask the latter portions of the action sequences that were not used (because the pole fell before that action took place), or not. We might want to mask it because that is not the "coding" part of the action sequence that represents the achieved fitness and what should be riffed on with mutations. 

From the other perspective, if we don't mask that regions, if we assume an x% mutation rate, mutations may happen proportionally less in the "coding region" of the sequence, in the region that reperesents the actions taken. (Although when mutation happens, it effects proportionally more of the relevant coding sequence, because it is shorter in L/R length.) This may be a beneficial regularization technique, or something like that.

Mutations in the "non-coding region", the latter part of the 200 L/R sequence that was not reached, can always be disgarded. But another benefit of having them is that the next generation, if it receives a mutation in it's coding region, may need those addional, still random at this point, L and R sequences, because it was able to keep the pole vertical a little while longer. 

Another reason we may not want to mask the non-coding action regions is that it just takes more work to implement and therefore does not follow a rule of parsimony.

So for the time being we will ignore masking the latter half of the action sequence before mutations. 

Note: the coding region of actions taken in the episode will still determine which action sequences survive to the next generation.

Note: We do some merging of evolutionary histories, starting with more parallel evolutions to start and then merging them together until there is just one combined generation at the end.

Note: The action sequence is our "agent" in some sense. But unlike a normal agent, [We don't need no observation; we don't need no thought control...](https://www.youtube.com/watch?v=bZwxTX2pWmw) We just need a fitness metric.

### evolve an action sequence to solve cart pole
```
uv run evo_cartpole.py evolve
```

### demo the best action sequence found
```
uv run evo_cartpole.py demo --sequence-file evo_cartpole_out/best_sequence.npy
```

#

# Part 2: Evolutionary Cart Pole w/ Bins

Now we discretize the initial starting state and learn a different action sequence for each bin. The state of the cart pole game consists of 4 floats.

At test time, we choose the action sequence from the bin the initial condition finds itself in.

The more bins we have, the better it performs across for any given initialization, yet it is not "generalizing" because it is just remembering a specific action sequence for each bin. The process is not getting a compressed representation (at least in its current form) of how to handle cart pole across many (initial) states. But what it does seem to excel at is finding discrete action sequences that have some fitness.

Note: If we increase the number of bins we used to split out the initial conditions from just 3, which only works ok at test time, to 10, the number of individual evolutionary processes we'll have to run, each comprised of many generations of mutating the action sequences of many action sequences, increasess from 3^4 = 81 to 10^4 = 10,000.

### Future Research #1: 

What if we "intelligently" guided the mutation process, the process that mutates (and mates?) action sequences with highest fitness scores at the end of each generation, by leveraging a neural net to decide which mutations to try? Could this make evolution quicker and our overall process more efficient?

### Future Research #2: 

Look into genetic algorithms, which from my understanding, is a subclass of evolutionary algorithms where the mutation happens at a lower level, say on the short programs that generates action sequences, not the action sequences themselves. In this model, we can still use the same fitness metric for the "phenotype" action sequence that is generated, but perhaps mutating the logical primitives is more fruitful than mutating the final action sequence themselves. Perhaps then we might have an "agent" on our hands, an agent that has a compressed model of it's environment and a tendency to take action sequences that result in high fitness, what evolution has selected for in terms of traits that lead to survival and reproduction. 

### Future Research #3: 
Combine idea #1 + #2

### Meta-research #4 (afterwards): 
Research work and results on evolutionary algorithms

### evolve 81 action sequencs (3**4 = 81 bins)
```
uv run evo_cartpole_bin.py evolve --generations 50 --grid-bins 3
```

### demo the best action sequence found (for the closest initial condition trained/evolved on)
```
uv run evo_cartpole_bin.py demo --sequence-file evo_cartpole_out/best_sequences_grid.npy --seed 123 --grid-bins 3
```