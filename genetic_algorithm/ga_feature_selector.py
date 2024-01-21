import numpy as np
import pandas as pd

import logging

from typing import Callable, List, Tuple

from multiprocessing import cpu_count, Pool
from numbers import Integral
from numpy.typing import ArrayLike
from copy import deepcopy
from enum import Enum

logger = logging.getLogger()
logger.setLevel(logging.INFO)

"""
Genetic Algorithm for Feature Selection
"""


class SelectionType(Enum):
    TOURNAMENT = "tournament"
    WHEEL = "roulette"

    @classmethod
    def default(cls):
        return cls.TOURNAMENT


class CrossoverType(Enum):
    ONE_POINT = "one_point"
    UNIFORM = "uniform"

    @classmethod
    def default(cls):
        return cls.UNIFORM


class MutationType(Enum):
    ONE_BIT = "one_bit"
    MULTI_BIT = "multi_bit"

    @classmethod
    def default(cls):
        return cls.ONE_BIT


class GAFeatureSelector:
    def __init__(
        self,
        estimator: object,
        fitness_func: Callable,
        population_size: int = 100,
        n_generations: int = 1000,
        selection_type: SelectionType = SelectionType.default(),
        crossover_type: CrossoverType = CrossoverType.default(),
        mutation_type: MutationType = MutationType.default(),
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
        tournament_size: int = 2,
        alpha: float = 0.99,
        n_jobs: int = 1,
        random_state: int = None,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.fitness_func = fitness_func
        self.alpha = alpha
        self.population_size = population_size
        self.verbose = verbose
        self.elite_size = 3

        if not isinstance(selection_type, SelectionType):
            raise ValueError(
                "selection_type must be an instance of SelectionType, got %s"
                % selection_type
            )
        self.selection_type = selection_type

        if not isinstance(crossover_type, CrossoverType):
            raise ValueError(
                "crossover_type must be an instance of CrossoverType, got %s"
                % crossover_type
            )
        self.crossover_type = crossover_type

        if not isinstance(mutation_type, MutationType):
            raise ValueError(
                "mutation_type must be an instance of MutationType, got %s"
                % mutation_type
            )
        self.mutation_type = mutation_type

        if n_generations <= 0:
            raise ValueError("n_generations must be > 0, got %d" % n_generations)
        self.n_generations = n_generations

        if not 0.0 < crossover_rate <= 1.0:
            raise ValueError(
                "crossover_rate must be in range (0, 1], got %f" % crossover_rate
            )
        self.crossover_rate = crossover_rate

        if not 0.0 < mutation_rate <= 1.0:
            raise ValueError(
                "mutation_rate must be in range (0, 1], got %f" % mutation_rate
            )
        self.mutation_rate = mutation_rate

        if tournament_size < 2:
            raise ValueError("tournament_size must be >= 2, got %d" % tournament_size)
        self.tournament_size = tournament_size

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs > 0 and n_jobs <= cpu_count():
            self.n_jobs = n_jobs
        else:
            raise ValueError(
                "n_jobs must be either -1 (use all CPUs) or positive integer \
                    less than your cpu cores number (got %d of possible %d)"
                % n_jobs,
                cpu_count(),
            )

        if isinstance(random_state, Integral):
            np.random.seed(random_state)

        self._best_chromosome = None
        self._best_fitness = np.inf
        self._best_history = []

    def _score_function(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        num_features: int,
    ) -> float:
        return (
            self.alpha * (1 - self.fitness_func(y_true, y_pred))
            + (1 - self.alpha) * num_features
        )

    def _init_population(self, num_features: int) -> List[np.array]:
        return [
            np.flatnonzero(np.random.choice([True, False], size=num_features))
            for _ in range(self.population_size)
        ]

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validate: pd.DataFrame,
        y_validate: pd.Series,
    ) -> tuple[np.array, float]:
        self.X_train = X_train
        self.y_train = y_train
        self.X_validate = X_validate
        self.y_validate = y_validate

        self._best_history = []
        self._best_chromosome = None
        self._best_fitness = np.inf

        num_features = X_train.shape[1]

        self.population = self._init_population(num_features)
        if self.verbose:
            logging.info("Initial population created")
        scores = self._evaluate(self.population)
        self._update_best_chromosome(self.population, scores)

        for generation in range(self.n_generations):
            elite = self._get_elite(self.population, scores)
            if self.verbose:
                logging.info("elite sizes:" + str([len(x) for x in elite]))
            if self.verbose:
                logging.info("Generation %d/%d", generation + 1, self.n_generations)
            selected_chromosomes = self._selection(
                self.population, scores, self.population_size - self.elite_size
            )
            if self.verbose:
                logging.info("Selection completed")
            crossovered_chromosomes = self._crossover(
                selected_chromosomes,
                num_features,
                self.population_size - self.elite_size,
            )
            if self.verbose:
                logging.info("Crossover completed")
            mutated_chromosomes = self._mutation(
                crossovered_chromosomes,
                num_features,
                self.population_size - self.elite_size,
            )
            if self.verbose:
                logging.info("Mutation completed")
            mutated_chromosomes.extend(elite)
            self.population = deepcopy(mutated_chromosomes)
            scores = self._evaluate(self.population)
            self._update_best_chromosome(self.population, scores)
        return self._best_chromosome, 1.0 - (
            self._best_fitness - (1 - self.alpha) * len(self._best_chromosome)
        )

    def _get_elite(
        self, population: List[np.array], fitness: np.array
    ) -> List[np.array]:
        elite_idxs = np.argsort(fitness)[: self.elite_size]
        return [population[idx] for idx in elite_idxs]

    def _estimate_fitness(self, chromosome: np.array) -> float:
        data = self.X_train.iloc[:, chromosome]

        estimator = self.estimator()

        estimator.fit(data, self.y_train)
        y_pred = estimator.predict(self.X_validate.iloc[:, chromosome])
        return self._score_function(self.y_validate, y_pred, len(chromosome))

    def _evaluate(self, population: List[np.array]) -> np.array:
        pool = Pool(self.n_jobs)
        fitness = pool.map(self._estimate_fitness, population)
        return np.array(fitness)

    def _update_best_chromosome(
        self, population: List[np.array], fitness: np.array
    ) -> None:
        best_chromosome_idx = np.argmin(fitness)
        best_chromosome = population[best_chromosome_idx]
        best_fitness = fitness[best_chromosome_idx]
        if self._best_chromosome is None or best_fitness < self._best_fitness:
            self._best_chromosome = best_chromosome
            self._best_fitness = best_fitness
            if self.verbose:
                logging.info(
                    f"Best chromosome is updated, fitness: {self._best_fitness},"
                    + f"features number: {len(self._best_chromosome)}"
                )
        elif self.verbose:
            logging.info(
                f"Current best chromosome (fitness: {best_fitness}, "
                + f"feature_num: {len(best_chromosome)}) is worse than the best one."
            )
        original_score = 1.0 - (
            self._best_fitness - (1 - self.alpha) * len(self._best_chromosome)
        )
        self._best_history.append(
            (self._best_chromosome, len(self._best_chromosome), original_score)
        )

    def _selection(
        self, population: List[np.array], fitness: np.array, population_size: int
    ) -> List[np.array]:
        if self.selection_type == SelectionType.TOURNAMENT:
            return self._tournament_selection(population, fitness, population_size)
        elif self.selection_type == SelectionType.WHEEL:
            return self._wheel_selection(population, fitness, population_size)

    def _tournament_selection(
        self, population: List[np.array], fitness: np.array, population_size: int
    ) -> List[np.array]:
        selected_chromosomes = []
        for _ in range(population_size):
            tournament_idxs = np.random.choice(
                population_size, size=self.tournament_size, replace=False
            )
            winner_idx = np.argmin(fitness[tournament_idxs])
            selected_chromosomes.append(population[tournament_idxs[winner_idx]])
        return selected_chromosomes

    def _wheel_selection(
        self, population: List[np.array], fitness: np.array, population_size: int
    ) -> List[np.array]:
        selected_chromosomes = []
        reversed_fitness = 1 / fitness
        cumulated_fitness = np.cumsum(reversed_fitness)
        for _ in range(population_size):
            rand_val = np.random.uniform(0, cumulated_fitness[-1])
            selected_chromosomes.append(
                population[np.searchsorted(cumulated_fitness, rand_val)]
            )
        return selected_chromosomes

    def _crossover(
        self, population: List[np.array], total_features: int, population_size: int
    ) -> List[np.array]:
        crossovers_num = np.random.binomial(population_size // 2, self.crossover_rate)
        selected_chromosomes = np.random.choice(
            population_size, size=crossovers_num * 2, replace=False
        )
        new_population = deepcopy(population)
        for i in range(0, len(selected_chromosomes), 2):
            parent_1 = population[selected_chromosomes[i]]
            parent_2 = population[selected_chromosomes[i + 1]]

            child_1, child_2 = [], []
            if self.crossover_type == CrossoverType.ONE_POINT:
                child_1, child_2 = self._one_point_crossover(
                    parent_1, parent_2, total_features
                )
            elif self.crossover_type == CrossoverType.UNIFORM:
                child_1, child_2 = self._uniform_crossover(
                    parent_1, parent_2, total_features
                )

            if child_1.shape[0] != 0:
                new_population[selected_chromosomes[i]] = child_1
            if child_2.shape[0] != 0:
                new_population[selected_chromosomes[i + 1]] = child_2
        return new_population

    def _one_point_crossover(
        self, parent_1: np.array, parent_2: np.array, total_features: int
    ) -> Tuple[np.array, np.array]:
        crossover_point = np.random.randint(1, total_features - 1)
        child_1 = np.concatenate(
            (
                parent_1[parent_1 < crossover_point],
                parent_2[parent_2 >= crossover_point],
            )
        )
        child_2 = np.concatenate(
            (
                parent_2[parent_2 < crossover_point],
                parent_1[parent_1 >= crossover_point],
            )
        )
        return child_1, child_2

    def _uniform_crossover(
        self, parent_1: np.array, parent_2: np.array, total_features: int
    ) -> Tuple[np.array, np.array]:
        child_1, child_2 = [], []
        mask = np.random.choice([True, False], size=total_features)
        child_1 = np.sort(
            np.concatenate(
                (
                    parent_1[np.isin(parent_1, np.where(mask))],
                    parent_2[np.isin(parent_2, np.where(~mask))],
                )
            )
        )
        child_2 = np.sort(
            np.concatenate(
                (
                    parent_2[np.isin(parent_2, np.where(mask))],
                    parent_1[np.isin(parent_1, np.where(~mask))],
                )
            )
        )
        return child_1, child_2

    def _mutation(
        self, population: List[np.array], total_features: int, population_size: int
    ) -> List[np.array]:
        new_population = deepcopy(population)
        for chromo_idx in range(population_size):
            genes_to_mutate = []
            if self.mutation_type == MutationType.MULTI_BIT:
                num_mutations = np.random.binomial(total_features, self.mutation_rate)
                genes_to_mutate = np.random.choice(
                    total_features, size=num_mutations, replace=False
                )
            elif self.mutation_type == MutationType.ONE_BIT:
                if np.random.uniform(0, 1) > self.mutation_rate:
                    continue
                genes_to_mutate = [np.random.randint(0, total_features - 1)]

            for gen in genes_to_mutate:
                idx = np.searchsorted(new_population[chromo_idx], gen)
                if (
                    idx < new_population[chromo_idx].shape[0]
                    and new_population[chromo_idx][idx] == gen
                ):
                    new_population[chromo_idx] = np.delete(
                        new_population[chromo_idx], idx
                    )
                else:
                    new_population[chromo_idx] = np.insert(
                        new_population[chromo_idx], idx, gen
                    )
        return new_population

    def get_best_history(self) -> List[tuple[np.array, int, float]]:
        return self._best_history
