import torch
import torch.nn as nn
from tqdm import tqdm
import asyncio
import pickle
import numpy as np

def optimize(swarm, evaluator, num_iter=100, lr=1e-1, display_freq=10, batch_size=4, record=False, experiment_id='experiment', use_learned_order=False):
    optimizer = torch.optim.Adam(swarm.connection_dist.parameters(), lr=lr)
    pbar = tqdm(range(num_iter), desc="Optimization Progress")
    utilities = []
    loop = asyncio.get_event_loop()
    for step in pbar:
        print(f"\n{'#'*80}")
        print(f"# ITERATION {step + 1}/{num_iter}")
        print(f"{'#'*80}")
        
        evaluator.reset()
        optimizer.zero_grad()
        tasks = []
        log_probs = []
        
        print(f"\nProcessing batch of {batch_size} puzzles...")
        for i in range(batch_size):
            _graph, log_prob = swarm.connection_dist.realize(swarm.composite_graph, use_learned_order=use_learned_order)
            tasks.append(evaluator.evaluate(_graph, return_moving_average=True))
            log_probs.append(log_prob)
        
        results = loop.run_until_complete(asyncio.gather(*tasks))
        utilities.extend([result[0] for result in results])
        if step == 0:
            moving_averages = np.array([np.mean(utilities) for _ in range(batch_size)])
        else:
            moving_averages = np.array([result[1] for result in results])
        loss = (-torch.stack(log_probs) * torch.tensor(np.array(utilities[-batch_size:]) - moving_averages)).mean()
        loss.backward()
        optimizer.step()

        # Print iteration summary
        current_utilities = utilities[-batch_size:]
        print(f"\n{'='*80}")
        print(f"ITERATION {step + 1} SUMMARY:")
        print(f"  Average Utility: {np.mean(current_utilities):.3f}")
        print(f"  Std Deviation:   {np.std(current_utilities):.3f}")
        print(f"  Min Score:       {np.min(current_utilities):.3f}")
        print(f"  Max Score:       {np.max(current_utilities):.3f}")
        print(f"  Total Puzzles:   {len(utilities)}")
        print(f"{'='*80}\n")
        
        # Update progress bar description
        pbar.set_postfix({
            'avg_utility': f'{np.mean(current_utilities):.3f}',
            'std': f'{np.std(current_utilities):.3f}'
        })

        if step % display_freq == display_freq - 1:
            if record:
                with open(f"result/crosswords/{experiment_id}_utilities_{step}.pkl", "wb") as file:
                    pickle.dump(utilities, file)
                torch.save(swarm.connection_dist.state_dict(), f"result/crosswords/{experiment_id}_edge_logits_{step}.pt")
