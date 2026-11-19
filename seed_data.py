"""Makes bookings.csv: 3,000 made-up room bookings at the Harbor View Inn.

Ready to use; you don't need to run it. The CSV it made is already in this folder,
and the lessons, the tests and report.py all read that file.

Every number in the file is made up, and the pattern is built in on purpose, so you
can check whether a model finds it. A real dataset's pattern is what you're trying
to find; this one is planted. Each booking's chance of being canceled comes from a
score, turned into a chance between 0 and 1 by the logistic curve:

    score = -2.3
            + 0.008 for each day between booking and arrival (lead_time_days)
            - 1.0 if the guest paid a deposit (deposit = 1)
            + 1.6 if it was booked on a travel site (channel = "travel_site")
            - 2.5 if the guest has stayed before (repeat_guest = 1)
            - 0.6 for each special request (special_requests)
            + 2.6 more if it was booked over 90 days ahead with no deposit
    chance of canceling = 1 / (1 + e^(-score))

The last line is the one that matters for this chapter: a long lead time and no
deposit together raise the chance more than the two do one at a time. A tree can
find a pattern like that by itself; a plain logistic regression adds each
feature's effect on its own and can't. About a quarter of the bookings are
canceled. The number of nights and guests have no effect at all.

The random numbers come from a fixed seed, 404, so this script makes the same file
every time on the same version of NumPy. NumPy doesn't promise the same random
numbers from a seed across its versions (NEP 19), so the CSV in this folder, not
this script, is what keeps every learner's numbers the same.
"""
import numpy as np
import pandas as pd

SEED = 404
BOOKINGS = 3000


def make_bookings(seed=SEED, count=BOOKINGS):
    """Return the made-up bookings as a DataFrame, one row per booking."""
    rng = np.random.default_rng(seed)
    lead_time_days = rng.gamma(1.6, 45, size=count).round().clip(0, 365)
    nights = rng.choice([1, 2, 3, 4, 5, 6, 7], size=count, p=[0.25, 0.3, 0.2, 0.1, 0.06, 0.04, 0.05])
    guests = rng.choice([1, 2, 3, 4], size=count, p=[0.2, 0.55, 0.12, 0.13])
    deposit = (rng.random(count) < 0.45).astype(int)
    channel = rng.choice(["direct", "phone", "travel_site"], size=count, p=[0.35, 0.2, 0.45])
    repeat_guest = (rng.random(count) < 0.15).astype(int)
    special_requests = rng.choice([0, 1, 2, 3], size=count, p=[0.5, 0.3, 0.15, 0.05])

    score = (-2.3 + 0.008 * lead_time_days - 1.0 * deposit + 1.6 * (channel == "travel_site")
             - 2.5 * repeat_guest - 0.6 * special_requests
             + 2.6 * ((lead_time_days > 90) & (deposit == 0)))
    chance = 1 / (1 + np.exp(-score))
    canceled = (rng.random(count) < chance).astype(int)

    return pd.DataFrame({
        "booking_id": [f"HV-{10001 + i}" for i in range(count)],
        "lead_time_days": lead_time_days.astype(int),
        "nights": nights.astype(int),
        "guests": guests.astype(int),
        "deposit": deposit,
        "channel": channel,
        "repeat_guest": repeat_guest,
        "special_requests": special_requests.astype(int),
        "canceled": canceled,
    })


if __name__ == "__main__":
    bookings = make_bookings()
    bookings.to_csv("bookings.csv", index=False)
    print("Wrote bookings.csv:", len(bookings), "bookings.")
