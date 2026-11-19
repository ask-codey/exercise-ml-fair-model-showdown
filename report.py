# Runs the whole showdown with your four functions from showdown.py: splits the Harbor View Inn's 3,000 bookings,
# searches a tree's settings on the training rows, compares four models on the same folds, and scores the chosen
# model once on the test rows. It prints what each step gave. Ready to use.
# Run it with: python report.py
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from showdown import build_pipeline, compare_models, final_report, make_search, split_bookings, tree_grid

bookings = pd.read_csv("bookings.csv")
X_train, X_test, y_train, y_test = split_bookings(bookings, 42)
print("1) bookings:", len(bookings))
print("2) training bookings:", len(X_train))
print("3) test bookings:", len(X_test))

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
grid = tree_grid([2, 3, 4, 5, 6, 8, None], [1, 10, 30])
search = make_search(build_pipeline(DecisionTreeClassifier(random_state=0)), grid, folds)
search.fit(X_train, y_train)
print("4) settings tried:", len(search.cv_results_["params"]))
print("5) best settings:", search.best_params_)

models = {
    "baseline (most frequent)": build_pipeline(DummyClassifier(strategy="most_frequent")),
    "logistic regression": build_pipeline(LogisticRegression(), scale=True),
    "tuned tree": search.best_estimator_,
    "forest, depth 6": build_pipeline(RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)),
}
results = compare_models(models, X_train, y_train, folds)
for number, (name, scores) in enumerate(results.items(), start=6):
    print(f"{number}) {name}: mean {scores['mean']:.3f}, spread {scores['std']:.3f}")

report = final_report(search, X_test, y_test)
print("10) chosen settings:", report["best_params"])
print("11) cross-validated accuracy:", round(report["cv_accuracy"], 3))
print("12) test accuracy, scored once:", round(report["test_accuracy"], 3))
