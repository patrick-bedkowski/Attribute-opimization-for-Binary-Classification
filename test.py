import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from genetic_algorithm.ga_feature_selector import (
    GAFeatureSelector,
    SelectionType,
    CrossoverType,
    MutationType,
)
from sklearn.metrics import f1_score


def plot_history(history, selection_type, crossover_type, mutation_type):
    scores, features_num = [], []
    for item in history:
        scores.append(item[2])
        features_num.append(item[1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        "Config: " + selection_type + " " + crossover_type + " " + mutation_type
    )
    axes[0].plot(features_num, color="red", label="num_features")
    axes[0].set_xlabel("Generation number")
    axes[0].set_ylabel("Number of features")
    axes[0].legend(loc="upper right")
    axes[1].plot(scores, color="blue", label="score")
    axes[1].set_xlabel("Generation number")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend(loc="upper right")
    plt.savefig(f"./plots/{selection_type}_{crossover_type}_{mutation_type}.png")


NO_OF_FEATURES = 300

# Assuming 'farm-ads-vect' is in SVMlight format
X, y = load_svmlight_file("data/farm-ads-vect")

# Convert sparse matrix to DataFrame
X = pd.DataFrame.sparse.from_spmatrix(X).iloc[:, :NO_OF_FEATURES]

# Shuffle data
X["y"] = y
X = X.sample(frac=1).reset_index(drop=True)
y = X["y"]
X = X.drop("y", axis=1)

X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.2, random_state=42
)

ga = GAFeatureSelector(
    estimator=KNeighborsClassifier,
    fitness_func=f1_score,
    n_jobs=-1,
    verbose=True,
    population_size=50,
    n_generations=50,
    mutation_rate=0.1,
    crossover_rate=0.8,
    random_state=42,
    selection_type=SelectionType.TOURNAMENT,
    crossover_type=CrossoverType.UNIFORM,
    mutation_type=MutationType.MULTI_BIT,
)

best_chromo, best_score = ga.fit(X_train, y_train, X_val, y_val)
print(f"Best chromosome: {best_chromo}")

X_test, y_test = X_test.iloc[:, best_chromo], y_test
model = KNeighborsClassifier()
model.fit(X_train.iloc[:, best_chromo], y_train)
y_pred = model.predict(X_test)
print(f"Test F1 score: {f1_score(y_test, y_pred)}")

history = ga.get_best_history()

plot_history(
    history,
    ga.selection_type.value,
    ga.crossover_type.value,
    ga.mutation_type.value,
)
