import random
from deap import base, creator, tools, algorithms



def setup_deap(fitness_function, param_order, bounds_dict, config, seed=42):
    random.seed(seed)

    # Prevent re-creation of classes when running multiple times
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Register ONE attribute function per parameter
    for p in param_order:
        low, high = bounds_dict[p]
        toolbox.register(f"attr_{p}", random.uniform, low, high)

    # Build the individual using the attribute functions
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        tuple(toolbox.__getattribute__(f"attr_{p}") for p in param_order),
        n=1
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=config["tournament_size"])

    # Add clamp decorator after mate
    toolbox.decorate(
        "mate",
        tools.DeltaPenalty(
            lambda _: False, 0,
            lambda ind: clamp_individual(ind, param_order, bounds_dict)
        )
    )

    # Add clamp decorator after mutate
    toolbox.decorate(
        "mutate",
        tools.DeltaPenalty(
            lambda _: False, 0,
            lambda ind: clamp_individual(ind, param_order, bounds_dict)
        )
    )

    return toolbox


def clamp_individual(ind, param_order, bounds_dict):
    for i, p in enumerate(param_order):
        low, high = bounds_dict[p]
        ind[i] = max(low, min(ind[i], high))
    return ind

