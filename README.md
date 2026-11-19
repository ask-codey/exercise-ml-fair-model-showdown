# Exercise: Fair Model Showdown

Exercise code for the lesson **Exercise: Fair Model Showdown** in the Machine Learning with Python course on [ask-codey.com](https://www.ask-codey.com).

The Harbor View Inn wants to know which bookings are likely to be canceled, and it has 3,000 past bookings to learn from. You'll write four functions in `showdown.py` that set up a fair contest between models: a grid of settings for a decision tree, a grid search scored by cross-validation, a comparison that scores every model on the same folds, and a final report that scores the chosen model once on the test bookings. Tests call each function and check what it returns.

## What's here

- `bookings.csv`: the 3,000 bookings, our own made-up data: each booking's `booking_id`, `lead_time_days` (days between booking and arrival), `nights`, `guests`, `deposit` (1 if a deposit was paid), `channel` (`direct`, `phone` or `travel_site`), `repeat_guest` (1 if the guest has stayed before), `special_requests` and `canceled` (1 if it was canceled). The lessons, the tests and `report.py` all read this file. Ready to use.
- `seed_data.py`: the script that made `bookings.csv`, so you can see how the data was built and the pattern planted in it. You don't need to run it: NumPy doesn't promise the same random numbers from a seed on every version, so the CSV, not the script, is what keeps everyone's numbers the same. Ready to use.
- `showdown.py`: the four functions, `tree_grid`, `make_search`, `compare_models` and `final_report`, for you to finish. Each one stops with `NotImplementedError: Not built yet.` for now, and the docstring under each says what it should do, with an example. It also holds `FEATURES`, `LABEL`, `split_bookings` (holds back a fifth of the bookings for the final test) and `build_pipeline` (a pipeline whose steps are named `prep` and `model`), ready to use.
- `explore.ipynb`: a notebook for looking at the bookings and trying your functions as you write them. The tests don't read it. Ready to use, and yours to change.
- `report.py`: runs the whole showdown with your four functions and prints what each step gave, so you can see them at work. Ready to use.
- `tests/test_showdown.py`: tests that call your functions and check what comes back. You don't need to change this file.
- `tests/conftest.py`: gives each test a fresh copy of the 3,000 bookings from `bookings.csv`. Ready to use.
- `requirements.txt`: the packages this exercise needs, each at an exact version: scikit-learn, pandas, NumPy, ipykernel (so VS Code can run the notebook) and pytest. Ready to use.
- `pytest.ini`: tells pytest where the tests are and where to find `showdown.py`. Ready to use.
- `.gitignore`: tells Git to leave out the virtual environment and the files Python, pytest and the notebook make as they run. Ready to use.

## Before you start

The packages go in a virtual environment: a private set of packages for this folder only. In VS Code's terminal, in this folder, run these three commands.

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On a Mac (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- The first command makes the virtual environment, in a folder named `.venv`.
- The second turns it on for this terminal: the prompt starts with `(.venv)` from then on.
- The third installs the packages `requirements.txt` lists into it.

*Virtual Environments and pip* explains all three. The `.venv` folder is in `.gitignore`, so Git leaves it out: these commands make it again on any computer.

If PowerShell answers that running scripts is disabled on this system, run this once, then run the second command again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Each new terminal starts with the virtual environment off. If the prompt doesn't start with `(.venv)`, run the second command again. (VS Code often turns it on for you.)

To look at the bookings, open `explore.ipynb`, choose the `.venv` kernel with **Select Kernel** (top right), and run the cells.

## Check your work

```bash
python -m pytest
```

At first every test fails: that is expected, because each function stops with a `NotImplementedError` for now. Each function you finish turns its tests green. A failing test's message says what it called, what it expected and what came back, for example *tree_grid([2, 4], [1, 10]) should return {'model__max_depth': [2, 4], 'model__min_samples_leaf': [1, 10]}, but it returned {'max_depth': [2, 4], 'min_samples_leaf': [1, 10]}.*

No test checks how accurate a model is. The tests check the grid's names, the search's settings, that every model is scored on the same folds, and that the test bookings are only scored, never learned from: a step inside the tests writes down which bookings every model was fitted on.

Some tests train trees on the whole file, so the full run takes a few seconds.

To see your functions at work on the whole file, run:

```bash
python report.py
```

## The rules

- Keep each function's name and parameters as they are, and keep `FEATURES`, `LABEL`, `NUMBERS`, `split_bookings` and `build_pipeline` as they are.
- Use the folds you're given: every model in a comparison is scored on exactly those folds.
- Never fit anything on the test bookings: `final_report` only scores the model the search already refit.
- Return each answer; don't only print it.
- Don't change the tests, `conftest.py`, `bookings.csv`, `seed_data.py` or `report.py`.

## Tips

- Try each step in `explore.ipynb` first, on the real bookings, then move the lines that work into `showdown.py`.
- A pipeline's settings are named after the step, two underscores, then the setting: `model__max_depth`.
- `GridSearchCV` takes the pipeline and the grid first, then `cv=` and `scoring=` by name.
- `cross_val_score` gives back an array of fold scores: `.mean()` and `.std()` give the two numbers, and `float(...)` turns each into a plain float.
- After you change `showdown.py`, restart the notebook's kernel before you try it there, so the notebook imports your newest code.

## Hand it in

When it works, push your solution to a public repo of your own on GitHub, then paste that repo's address into **Submit your exercise** on the lesson page at [ask-codey.com](https://www.ask-codey.com). That's what completes the lesson. The lesson page shows the Git commands, step by step.
