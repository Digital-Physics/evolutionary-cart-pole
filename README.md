# Cart Pole

We use Evolutionary Algorithms to evolve a Left/Right action sequence solution.  
We assume the pole is initialized the same, so we'll use the same random seed.
This deterministic environment should help learning. We can relax this assumption later.  

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

# demo the best action sequence found
```
uv run evo_cartpole.py demo --sequence-file evo_cartpole_out/best_sequence.npy
```

