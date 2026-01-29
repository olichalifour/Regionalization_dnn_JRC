import os
import numpy as np
import pandas as pd
from deap import algorithms, tools

def optimize_for_catchment(toolbox, config, catchment_id, input_cols):
    pop = toolbox.population(n=config["population_size"])
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    verbose = config.get("verbose", True)
    all_history = []

    for gen in range(config["n_generations"]):
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)

        # Record generation statistics
        gen_fit = [ind.fitness.values[0] for ind in pop]
        gen_data = pd.DataFrame([ind for ind in pop], columns=input_cols)
        gen_data["pred_kge"] = [f for f in gen_fit]  # flip sign back
        gen_data["generation"] = gen
        all_history.append(gen_data)

        if verbose:
            print(f"Catchment {catchment_id} | Gen {gen+1}/{config['n_generations']} | "
                  f"Best KGE: {max(gen_data['pred_kge']):.3f} | "
                  f"Mean KGE: {gen_data['pred_kge'].mean():.3f}")

        # Select, clone, mate, mutate
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < config["crossover_prob"]:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < config["mutation_prob"]:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        pop[:] = offspring

    # Combine all history and order per kge
    history_df = pd.concat(all_history, ignore_index=True)
    history_df.sort_values("pred_kge",ascending=False,inplace=True)
    # Save catchment history
    os.makedirs("results/history", exist_ok=True)
    out_path = f"results/history/catchment_{catchment_id}_history.csv"
    history_df.to_csv(out_path, index=False)

    if verbose:
        print(f"Saved optimization history for catchment {catchment_id} → {out_path}")

    best_params = hof[0]
    best_fitness = hof[0].fitness.values[0]

    return best_params, best_fitness