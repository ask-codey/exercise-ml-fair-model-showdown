"""Harbor View Inn: four functions for you to finish.

The docstring under each function says what it should do, with examples.
Replace each raise NotImplementedError("Not built yet.") line with your own code.
Return each answer: the tests read what comes back, not what's printed.

bookings is a pandas table (a DataFrame) with one row per booking and the columns
booking_id, lead_time_days, nights, guests, deposit, channel, repeat_guest,
special_requests and canceled, like the one pd.read_csv("bookings.csv") gives you.
FEATURES, LABEL, split_bookings and build_pipeline are ready to use. The imports
below are everything the four functions need.
"""
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The columns the models look at (their features), in this order. Ready to use.
FEATURES = ["lead_time_days", "nights", "guests", "deposit", "channel", "repeat_guest", "special_requests"]

# The column the models learn to guess (their label): 1 if the booking was canceled, 0 if not. Ready to use.
LABEL = "canceled"

# The number columns, which pass straight through (or are standardized, with scale=True). Ready to use.
NUMBERS = ["lead_time_days", "nights", "guests", "deposit", "repeat_guest", "special_requests"]


def split_bookings(bookings, seed):
    """Ready to use. Hold back a fifth of the bookings for the final test, with the
    same share of canceled bookings on both sides, and return
    X_train, X_test, y_train, y_test. With the 3,000 bookings and seed 42:
    2,400 training bookings and 600 test bookings."""
    return train_test_split(bookings[FEATURES], bookings[LABEL], test_size=0.2,
                            stratify=bookings[LABEL], random_state=seed)


def build_pipeline(model, scale=False):
    """Ready to use. Return a pipeline with two steps, named "prep" and "model":
    prep one-hot encodes the channel column and passes the number columns through
    (or standardizes them, with scale=True, for a model that cares about scale,
    such as LogisticRegression), and model is the model you hand it."""
    numbers = StandardScaler() if scale else "passthrough"
    prep = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), ["channel"]), (numbers, NUMBERS))
    return Pipeline([("prep", prep), ("model", model)])


def tree_grid(depths, leaf_sizes):
    """Return the grid for a grid search over a pipeline's decision tree, the step named "model".

    The grid is a dictionary with two keys: "model__max_depth", holding the depths to try,
    and "model__min_samples_leaf", holding the leaf sizes to try, each as a list in the
    order it was given.

    tree_grid([2, 4], [1, 10]) -> {"model__max_depth": [2, 4], "model__min_samples_leaf": [1, 10]}
    """
    raise NotImplementedError("Not built yet.")


def make_search(pipeline, grid, folds):
    """Return a GridSearchCV that tries every setting in grid on pipeline, scored by accuracy.

    Use the pipeline and the grid as they're given, cv=folds (so the search uses exactly
    these folds) and scoring="accuracy". Leave every other setting at its default (refit
    stays True). Don't fit it: the caller fits it, on the training rows only.

    make_search(pipeline, grid, folds) -> a GridSearchCV whose estimator is pipeline,
    whose param_grid is grid, whose cv is folds and whose scoring is "accuracy",
    and which has no best_params_ yet.
    """
    raise NotImplementedError("Not built yet.")


def compare_models(models, X, y, folds):
    """Score every model on the same folds and return each one's mean and spread.

    models is a dictionary of a name to a model (or pipeline), not fitted. For each one,
    get its fold scores with cross_val_score(model, X, y, cv=folds, scoring="accuracy"),
    and return a dictionary of the same names, each to a dictionary with two numbers:
    "mean", the scores' mean (scores.mean()), and "std", their standard deviation
    (scores.std()), each as a plain float, not rounded.

    With ten bookings whose labels are 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, the folds
    KFold(n_splits=5), and models {"baseline": DummyClassifier(strategy="most_frequent")}:
    the fold scores are 1.0, 0.5, 1.0, 0.0 and 1.0, so the result is
    {"baseline": {"mean": 0.7, "std": 0.4}}
    """
    raise NotImplementedError("Not built yet.")


def final_report(search, X_test, y_test):
    """Return the chosen settings, their cross-validated score and the one test score.

    search is a GridSearchCV that has already been fitted on the training rows. Don't
    fit anything here: score the refit best model, which the search already holds, on
    the test rows once, with search.score(X_test, y_test). Return a dictionary:
    "best_params", the search's best_params_; "cv_accuracy", its best_score_ as a plain
    float; and "test_accuracy", the test score as a plain float. Nothing is rounded.

    With a search over {"model__max_depth": [1, 2]} for a pipeline with one step, a
    DecisionTreeClassifier(random_state=0) named "model", fitted on eight bookings with
    lead times 10, 20, 30, 40, 100, 110, 120 and 130 days, the first four not canceled and
    the last four canceled, and folds StratifiedKFold(n_splits=2, shuffle=True,
    random_state=0), then scored on four test bookings with lead times 50, 95, 105 and 60
    days, labeled 0, 1, 1 and 1:
    {"best_params": {"model__max_depth": 1}, "cv_accuracy": 1.0, "test_accuracy": 0.75}
    """
    raise NotImplementedError("Not built yet.")
