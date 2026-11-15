"""Landmark cohort and features available by the inclusive day-28 cutoff."""
from pathlib import Path
import pandas as pd
from .data import KEY, require

NUMERIC = ["studied_credits", "num_of_prev_attempts", "date_registration",
           "clicks", "active_days", "last_activity", "recent_clicks"]
CATEGORICAL = ["code_module"]
FEATURES = NUMERIC + CATEGORICAL
AUDIT = ["gender", "age_band", "disability", "imd_band"]


def build(root, tables, cutoff=28):
    require(cutoff >= 0, "Cutoff must be nonnegative")
    info = tables["studentInfo"]
    d = info.merge(tables["studentRegistration"], on=KEY, validate="one_to_one")
    # Unregistration is used solely to define who is still enrolled at prediction.
    eligible = d.date_registration.notna() & d.date_registration.le(cutoff)
    eligible &= d.date_unregistration.isna() | d.date_unregistration.gt(cutoff)
    d = d.loc[eligible].copy()
    parts = []
    for events in pd.read_csv(Path(root) / "studentVle.csv", chunksize=500_000, na_values=["?"]):
        events = events.loc[events.date.between(0, cutoff)].copy()
        events["recent_clicks"] = events.sum_click.where(events.date.gt(cutoff - 7), 0)
        parts.append(events.groupby(KEY + ["date"], as_index=False).agg(
            clicks=("sum_click", "sum"), recent_clicks=("recent_clicks", "sum")))
    daily = pd.concat(parts, ignore_index=True).groupby(KEY + ["date"], as_index=False).sum()
    agg = daily.groupby(KEY, as_index=False).agg(
        clicks=("clicks", "sum"), active_days=("date", "nunique"),
        last_activity=("date", "max"), recent_clicks=("recent_clicks", "sum"))
    d = d.merge(agg, on=KEY, how="left", validate="one_to_one")
    d[["clicks", "active_days", "recent_clicks"]] = d[["clicks", "active_days", "recent_clicks"]].fillna(0)
    d["last_activity"] = d.last_activity.fillna(-1)
    d["target"] = d.final_result.isin(["Fail", "Withdrawn"]).astype(int)
    # Deliberately omit assessment scores: OULAD has submission time but no grade-publication time.
    return d[KEY + AUDIT + FEATURES[:-1] + ["target", "final_result"]].reset_index(drop=True)


def temporal_split(d):
    train = d.loc[d.code_presentation.isin(["2013B", "2013J"])].copy()
    val = d.loc[d.code_presentation.eq("2014B") & ~d.id_student.isin(train.id_student)].copy()
    seen = set(train.id_student) | set(val.id_student)
    test = d.loc[d.code_presentation.eq("2014J") & ~d.id_student.isin(seen)].copy()
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        require(not part.empty and part.target.nunique() == 2, f"{name} needs both classes")
    require(not set(train.id_student) & set(test.id_student), "Student overlap")
    return train, val, test
