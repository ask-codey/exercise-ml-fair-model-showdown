# Tests for showdown.py. Run them with: python -m pytest (in the exercise folder, with the virtual environment
# active). Each test calls one of your functions, on the 3,000 bookings in bookings.csv or on a small table written
# in the test, and checks what comes back. A failing test says what it called, what it expected and what came back.
# None of them checks how accurate a model is: they check the grid, the search's settings, that every model is scored
# on the same folds, and that the test rows are only ever scored, never learned from.
# You don't need to change this file.
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from showdown import build_pipeline, compare_models, final_report, make_search, split_bookings, tree_grid

# Every time a RecordFit step is fitted, it adds the row labels it was fitted on to FITS, so a test can see which
# bookings each model learned from. It passes the table on unchanged.
FITS = []


class RecordFit(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        FITS.append(set(X.index))
        return self

    def transform(self, X):
        return X


def recording(model):
    """A pipeline with a RecordFit step first, then the same "prep" and "model" steps as build_pipeline."""
    pipeline = build_pipeline(model)
    return Pipeline([("record", RecordFit())] + pipeline.steps)


def check(call, got, expected):
    """Fails with a message that says what the call returned and what it should return."""
    assert got == expected, f"{call} should return {expected!r}, but it returned {got!r}."


# Ten bookings for the worked example in compare_models' docstring.
TEN = pd.DataFrame({"lead_time_days": [5, 12, 30, 41, 60, 75, 90, 120, 150, 200]})
TEN_LABELS = pd.Series([0, 0, 0, 1, 0, 0, 1, 1, 0, 0])

# Eight training bookings and four test bookings for the worked example in final_report's docstring.
EIGHT = pd.DataFrame({"lead_time_days": [10, 20, 30, 40, 100, 110, 120, 130]})
EIGHT_LABELS = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
FOUR = pd.DataFrame({"lead_time_days": [50, 95, 105, 60]})
FOUR_LABELS = pd.Series([0, 1, 1, 1])


def small_search():
    pipeline = Pipeline([("model", DecisionTreeClassifier(random_state=0))])
    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    search = make_search(pipeline, {"model__max_depth": [1, 2]}, folds)
    return search.fit(EIGHT, EIGHT_LABELS)


# tree_grid

def test_1_tree_grid_2_4_and_1_10_gives_model_max_depth_and_model_min_samples_leaf():
    check("tree_grid([2, 4], [1, 10])", tree_grid([2, 4], [1, 10]),
          {"model__max_depth": [2, 4], "model__min_samples_leaf": [1, 10]})


def test_2_tree_grid_keeps_none_and_the_order_given():
    check("tree_grid([8, None, 3], [30, 1])", tree_grid([8, None, 3], [30, 1]),
          {"model__max_depth": [8, None, 3], "model__min_samples_leaf": [30, 1]})


def test_3_tree_grid_names_settings_the_pipelines_model_step_has():
    grid = tree_grid([2, 4], [1, 10])
    pipeline = build_pipeline(DecisionTreeClassifier(random_state=0))
    for name, values in grid.items():
        assert name.startswith("model__"), \
            f"Each name in the grid should start with the step's name and two underscores, model__, but one was {name!r}."
        assert name in pipeline.get_params(), \
            f"The pipeline has no setting named {name!r}; check the spelling after model__."


# make_search

def test_4_make_search_returns_an_unfitted_grid_search_with_the_pipeline_and_grid_given():
    pipeline = build_pipeline(DecisionTreeClassifier(random_state=0))
    grid = tree_grid([2, 4], [1, 10])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = make_search(pipeline, grid, folds)
    assert isinstance(search, GridSearchCV), f"make_search should return a GridSearchCV, but it returned {search!r}."
    assert search.estimator is pipeline, "The search's estimator should be the pipeline make_search was given."
    check("the search's param_grid", search.param_grid, grid)
    assert not hasattr(search, "best_params_"), \
        "make_search should return the search without fitting it: the caller fits it on the training rows."


def test_5_make_search_uses_the_folds_given_and_scores_by_accuracy_and_refits():
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = make_search(build_pipeline(DecisionTreeClassifier(random_state=0)), tree_grid([2], [1]), folds)
    assert search.cv is folds, f"The search's cv should be the folds make_search was given, but it was {search.cv!r}."
    check("the search's scoring", search.scoring, "accuracy")
    check("the search's refit", search.refit, True)


def test_6_make_search_fitted_on_training_rows_picks_depth_4_and_leaf_1_from_the_csv(bookings):
    X_train, X_test, y_train, y_test = split_bookings(bookings, 42)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = make_search(build_pipeline(DecisionTreeClassifier(random_state=0)),
                         tree_grid([2, 3, 4, 5, 6, 8, None], [1, 10, 30]), folds)
    search.fit(X_train, y_train)
    check("len(search.cv_results_['params']) for 7 depths and 3 leaf sizes", len(search.cv_results_["params"]), 21)
    check("search.best_params_", search.best_params_, {"model__max_depth": 4, "model__min_samples_leaf": 1})


# compare_models

def test_7_compare_models_ten_bookings_baseline_gives_mean_0_7_and_std_0_4():
    result = compare_models({"baseline": DummyClassifier(strategy="most_frequent")}, TEN, TEN_LABELS,
                            KFold(n_splits=5))
    check("the names compare_models returns", list(result), ["baseline"])
    check("the keys for one model", sorted(result["baseline"]), ["mean", "std"])
    assert result["baseline"]["mean"] == pytest.approx(0.7), \
        f"The baseline's mean should be 0.7, but it was {result['baseline']['mean']!r}."
    assert result["baseline"]["std"] == pytest.approx(0.4), \
        f"The baseline's std should be 0.4, but it was {result['baseline']['std']!r}."


def test_8_compare_models_returns_plain_floats():
    result = compare_models({"baseline": DummyClassifier(strategy="most_frequent")}, TEN, TEN_LABELS,
                            KFold(n_splits=5))
    for key in ["mean", "std"]:
        assert type(result["baseline"][key]) is float, \
            f"result['baseline'][{key!r}] should be a plain float, but it was {type(result['baseline'][key]).__name__}."


def test_9_compare_models_matches_cross_val_score_for_a_tree_and_a_baseline(bookings):
    X_train, X_test, y_train, y_test = split_bookings(bookings, 42)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    models = {
        "baseline": build_pipeline(DummyClassifier(strategy="most_frequent")),
        "tree, depth 4": build_pipeline(DecisionTreeClassifier(max_depth=4, random_state=0)),
    }
    result = compare_models(models, X_train, y_train, folds)
    check("the names compare_models returns", list(result), ["baseline", "tree, depth 4"])
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="accuracy")
        assert result[name]["mean"] == pytest.approx(scores.mean()), \
            f"{name}: the mean should be {float(scores.mean())!r}, but it was {result[name]['mean']!r}."
        assert result[name]["std"] == pytest.approx(scores.std()), \
            f"{name}: the std should be {float(scores.std())!r}, but it was {result[name]['std']!r}."


def test_10_compare_models_trains_every_model_on_the_same_five_folds(bookings):
    X_train, X_test, y_train, y_test = split_bookings(bookings, 42)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    models = {
        "baseline": recording(DummyClassifier(strategy="most_frequent")),
        "tree, depth 4": recording(DecisionTreeClassifier(max_depth=4, random_state=0)),
    }
    FITS.clear()
    compare_models(models, X_train, y_train, folds)
    expected = [set(X_train.index[train]) for train, _ in folds.split(X_train, y_train)]
    check("how many times the two models were fitted (5 folds each)", len(FITS), 10)
    assert FITS[:5] == expected, "The baseline wasn't trained on the folds compare_models was given."
    assert FITS[5:] == expected, "The tree wasn't trained on the same folds as the baseline."


# final_report

def test_11_final_report_eight_bookings_gives_depth_1_cv_1_0_and_test_0_75():
    check("final_report(search, four test bookings)", final_report(small_search(), FOUR, FOUR_LABELS),
          {"best_params": {"model__max_depth": 1}, "cv_accuracy": 1.0, "test_accuracy": 0.75})


def test_12_final_report_returns_plain_floats():
    report = final_report(small_search(), FOUR, FOUR_LABELS)
    for key in ["cv_accuracy", "test_accuracy"]:
        assert type(report[key]) is float, f"report[{key!r}] should be a plain float, but it was {type(report[key]).__name__}."


def test_13_final_report_scores_the_refit_best_model_and_fits_nothing(bookings):
    X_train, X_test, y_train, y_test = split_bookings(bookings, 42)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(recording(DecisionTreeClassifier(random_state=0)), tree_grid([2, 4], [1, 10]),
                          cv=folds, scoring="accuracy").fit(X_train, y_train)
    expected = (search.best_estimator_.predict(X_test) == y_test).mean()
    FITS.clear()
    report = final_report(search, X_test, y_test)
    check("how many times anything was fitted inside final_report", len(FITS), 0)
    assert report["test_accuracy"] == pytest.approx(expected), \
        f"test_accuracy should be the refit best model's accuracy on the test rows, {float(expected)!r}, but it was {report['test_accuracy']!r}."
    check("report['best_params']", report["best_params"], search.best_params_)
    assert report["cv_accuracy"] == pytest.approx(search.best_score_), \
        f"cv_accuracy should be the search's best_score_, {float(search.best_score_)!r}, but it was {report['cv_accuracy']!r}."


def test_14_the_whole_showdown_never_fits_anything_on_the_600_test_bookings(bookings):
    X_train, X_test, y_train, y_test = split_bookings(bookings, 42)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    FITS.clear()
    search = make_search(recording(DecisionTreeClassifier(random_state=0)), tree_grid([2, 4], [1, 10]), folds)
    search.fit(X_train, y_train)
    compare_models({"baseline": recording(DummyClassifier(strategy="most_frequent")),
                    "tuned tree": search.best_estimator_}, X_train, y_train, folds)
    final_report(search, X_test, y_test)
    test_rows = set(X_test.index)
    leaked = [len(rows & test_rows) for rows in FITS if rows & test_rows]
    check("how many fits learned from any of the 600 test bookings", len(leaked), 0)
    assert len(FITS) > 0, "Nothing was fitted at all: make_search's search should be fitted on the training rows."
