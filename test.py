import pandas as pd
import numpy as np
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
from sklearn.metrics import f1_score, accuracy_score


def plot_history(history, selection_type, crossover_type, mutation_type, frame_size):
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
    plt.savefig(
        f"./plots/{selection_type}_{crossover_type}_{mutation_type}_{frame_size}.png"
    )


NO_OF_FEATURES = 1000

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
    verbose=True,
    population_size=50,
    n_generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    # random_state=42,
    selection_type=SelectionType.TOURNAMENT,
    crossover_type=CrossoverType.UNIFORM,
    mutation_type=MutationType.ONE_BIT,
)

best_chromo, best_score = ga.fit(X_train, y_train, X_val, y_val)
print(f"Best chromosome: {best_chromo}")

X_temp_test, y_temp_test = X_test.iloc[:, best_chromo], y_test
model = KNeighborsClassifier()
model.fit(X_train.iloc[:, best_chromo], y_train)
y_pred = model.predict(X_temp_test)
test_score = f1_score(y_temp_test, y_pred)
print(f"Test score: {test_score}")
acc = accuracy_score(y_temp_test, y_pred)
print(f"Accuracy: {acc}")

history = ga.get_best_history()
scores, features_num = [], []
for item in history:
    scores.append(item[2])
    features_num.append(item[1])

df = pd.DataFrame({"num_features": features_num, "f1_score": scores})
df.to_csv("history.csv")
plot_history(
    history,
    ga.selection_type.value,
    ga.crossover_type.value,
    ga.mutation_type.value,
    1000,
)
