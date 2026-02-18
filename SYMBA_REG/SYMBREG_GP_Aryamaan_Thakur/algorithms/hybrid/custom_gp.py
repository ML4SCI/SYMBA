import math
import numpy as np
import operator
from deap import base, creator, tools, gp, algorithms
from dataclasses import dataclass, field, asdict
from typing import Optional

UPPER_LIMIT = 1e7

def r2_score_from_fitness(points, fitness_score):
    """
    Calculates the R^2 score based on the fitness score of an individual.
    
    Args:
        points (list of tuples): Data points used in the evaluation.
        fitness_score (float): The fitness score of the individual.
    
    Returns:
        float: The R^2 score, indicating the goodness of fit.
    """
    TSS = 0.0
    mean_y = sum([y for _, y in points]) / len(points)
    for x, y in points:
        TSS += (y - mean_y) ** 2
    r2_score = 1 - (float(fitness_score) / TSS)
    return r2_score

# Protected operations to handle exceptions gracefully in genetic programming
def protected_div(x1, x2):
    """
    Protected division to prevent division by zero.
    """
    try:
        return x1 / x2
    except ZeroDivisionError:
        return UPPER_LIMIT

def protected_exp(x1):
    """
    Protected exponential to handle overflow by setting a maximum bound.
    """
    try:
        return math.exp(x1) if x1 < 100 else 0.0
    except OverflowError:
        return UPPER_LIMIT

def protected_log(x1):
    """
    Protected logarithm to handle domain errors gracefully.
    """
    try:
        return math.log(x1)
    except:
        return -UPPER_LIMIT

def protected_sqrt(x1):
    """
    Protected square root to handle negative inputs.
    """
    try:
        return math.sqrt(x1)
    except:
        return 0

def protected_pow(x1, x2):
    """
    Protected power function to handle overflow and other errors.
    """
    try:
        a = math.pow(x1, x2)
        return a
    except:
        return UPPER_LIMIT

@dataclass
class CustomGPConfig:
    """
    Configuration class for Custom Genetic Programming (GP) parameters.
    
    Attributes:
        pop_size (int): Population size for the GP.
        cxpb (float): Crossover probability.
        mutpb (float): Mutation probability.
        num_generations (int): Number of generations for the GP.
        num_vars (int): Number of variables in the symbolic regression problem.
    """
    pop_size: Optional[int] = field(default=500)
    cxpb: Optional[float] = field(default=0.7)
    mutpb: Optional[float] = field(default=0.1)
    num_generations: Optional[int] = field(default=15)
    num_vars: Optional[int] = field(default=5)
    verbose: Optional[bool] = field(default=False)

class CustomGP:
    """Custom Genetic Programming class for symbolic regression tasks."""
    def __init__(self, config):
        """
        Initializes the CustomGP instance with a given configuration.

        Args:
            config (CustomGPConfig): Configuration for the genetic programming algorithm.
        """
        self.config = config
        self.pset = self.get_pset(config.num_vars)

    @staticmethod
    def get_pset(num_vars):
        pset = gp.PrimitiveSet("MAIN", num_vars)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(math.sin, 1)
        pset.addPrimitive(math.cos, 1)
        pset.addPrimitive(math.tan, 1)
        pset.addPrimitive(math.tanh, 1)
        for i in range(1, 5):
            pset.addTerminal(int(i))
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(protected_pow, 2)
        pset.addPrimitive(protected_exp, 1)
        pset.addPrimitive(protected_log, 1)
        pset.addPrimitive(protected_sqrt, 1)
        pset.addTerminal(math.pi, name="pi")
        pset.addTerminal(math.e, name="E")
        rename_kwargs = {"ARG{}".format(i): f"s_{i+1}" for i in range(0, num_vars)}
        pset.renameArguments(**rename_kwargs)
        return pset


    def get_toolbox(self, points):
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=4)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", self.get_initial_population, toolbox=toolbox)
        toolbox.register("compile", gp.compile, pset=self.pset)
        toolbox.register("evaluate", self.evalSymbReg, points=points, pset=self.pset)
        toolbox.register("select", tools.selAutomaticEpsilonLexicase)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=self.pset)
        return toolbox

    @staticmethod
    def get_initial_population(pop_size, candidates, pset, toolbox):
        population = []
        
        for expr in candidates:
            try:
                candidate = creator.Individual.from_string(expr, pset)
                population.append(candidate)
            except:
                continue
        
        for i in range(pop_size - len(population)):
            random_candidate = toolbox.individual()
            population.append(random_candidate)
        return population

    @staticmethod
    def evalSymbReg(individual, points, pset):
        func = gp.compile(expr=individual, pset=pset)
        sqerrors = (((func(*x) - y)**2)/len(points) for x, y in points)
        return math.fsum(sqerrors),

    def register_stats(self):
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        return stats
    
    def __call__(self, points, candidate_equations):
        """
        Runs the genetic programming process and returns the best individual and R2 score.

        Args:
            points (list): List of tuples, where each tuple is a (features, target) pair.
            candidate_equations (list): List of candidate equations for initialization.

        Returns:
            tuple: The best individual and corresponding R2 score.
        """
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        toolbox = self.get_toolbox(points)
        population = toolbox.population(self.config.pop_size, candidate_equations, self.pset)

        fitness_results = [self.evalSymbReg(individual, points, self.pset) for individual in population]

        for individual, fitness_score in zip(population, fitness_results):
            individual.fitness.values = fitness_score
            if float(fitness_score[0]) < 1e-7:
                r2_score = r2_score_from_fitness(points, fitness_score[0])
                if self.config.verbose:
                    print("R2_score:", r2_score)
                    print("Fitness:", fitness_score[0])
                return individual, r2_score

        hof = tools.HallOfFame(5)
        stats = self.register_stats()
        population, log = algorithms.eaSimple(
            population,
            toolbox,
            self.config.cxpb,
            self.config.mutpb,
            self.config.num_generations,
            stats=stats,
            halloffame=hof,
            verbose=self.config.verbose
        )

        r2_score = r2_score_from_fitness(points, hof[0].fitness.values[0])
        if self.config.verbose:
            print("R2_score:", r2_score)
            print("Fitness:", hof[0].fitness.values)
        return hof[0], r2_score
