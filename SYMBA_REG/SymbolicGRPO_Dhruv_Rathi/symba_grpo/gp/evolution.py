"""Genetic-programming search shared by both refinement arms."""

from __future__ import annotations

import operator
from typing import List, Optional, Tuple

import numpy as np
from deap import algorithms, base, creator, gp, tools


MAX_TREE_HEIGHT = 12


def _ensure_creator() -> None:
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


def setup_toolbox(pset: gp.PrimitiveSet, points: list) -> base.Toolbox:
    """Construct the frozen epsilon-lexicase GP configuration."""
    _ensure_creator()
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    from symba_grpo.gp.fast_eval import make_vectorized_evaluator

    toolbox.register("evaluate", make_vectorized_evaluator(points))
    toolbox.register("select", tools.selAutomaticEpsilonLexicase)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))
    return toolbox


def _r2_from_fitness(points: list, fitness: float) -> float:
    mean_y = sum(y for _, y in points) / len(points)
    variance = sum((y - mean_y) ** 2 for _, y in points) / len(points)
    if variance == 0:
        return 1.0 if fitness < 1e-10 else 0.0
    return 1.0 - fitness / variance


def run_gp(
    toolbox: base.Toolbox,
    points: list,
    pset: gp.PrimitiveSet,
    seed_exprs: Optional[List[str]] = None,
    pop_size: int = 500,
    num_generations: int = 30,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    verbose: bool = False,
) -> Tuple:
    """Run the GP search and return its best expression and population."""
    _ensure_creator()
    population = (
        seed_population(pop_size, seed_exprs, pset, toolbox)
        if seed_exprs
        else toolbox.population(n=pop_size)
    )
    for individual in population:
        if not individual.fitness.valid:
            individual.fitness.values = toolbox.evaluate(individual)
    convergence = [(0, min(ind.fitness.values[0] for ind in population))]
    hall = tools.HallOfFame(5)
    hall.update(population)
    # Early stopping is a threshold decision, not a license to return the
    # first qualifying individual in population order.  Always return the
    # best member represented by the Hall of Fame.
    if hall and hall[0].fitness.values[0] < 1e-7:
        best = hall[0]
        return best, _r2_from_fitness(points, best.fitness.values[0]), population, hall, convergence

    statistics = tools.Statistics(lambda individual: individual.fitness.values)
    statistics.register("avg", np.mean)
    statistics.register("min", np.min)
    population, log = algorithms.eaSimple(
        population,
        toolbox,
        cxpb,
        mutpb,
        num_generations,
        stats=statistics,
        halloffame=hall,
        verbose=verbose,
    )
    for record in log:
        convergence.append((record["gen"], float(np.asarray(record["min"]).reshape(-1)[0])))
    best = hall[0]
    return best, _r2_from_fitness(points, best.fitness.values[0]), population, hall, convergence


def seed_population(pop_size: int, seed_exprs: List[str], pset: gp.PrimitiveSet, toolbox: base.Toolbox) -> list:
    """Insert accepted transformer trees and fill the remaining fixed population randomly."""
    from loguru import logger

    _ensure_creator()
    # Draw the entire matched random population first. Accepted proposals then
    # replace a prefix, so different admission counts cannot shift the GP RNG
    # stream or the common random suffix.
    population = toolbox.population(n=pop_size)
    accepted = []
    skipped = 0
    for expression in seed_exprs:
        try:
            expression = expression.strip()
            if expression.lstrip("-").isdigit():
                skipped += 1
                continue
            individual = creator.Individual.from_string(expression, pset)
            if individual.height > MAX_TREE_HEIGHT:
                skipped += 1
                continue
            accepted.append(individual)
        except Exception:
            skipped += 1
    if skipped:
        logger.info("seed_population: {} seeded, {} skipped", min(len(accepted), pop_size), skipped)
    count = min(len(accepted), pop_size)
    population[:count] = accepted[:count]
    return population
