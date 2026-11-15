"""Authenticated public-source acquisition, checksums and relational validation."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/349/open%2Buniversity%2Blearning%2Banalytics%2Bdataset.zip"
EXPECTED_SHA256 = "f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4"
KEY = ["code_module", "code_presentation", "id_student"]
FILES = ["courses", "assessments", "studentInfo", "studentRegistration",
         "studentAssessment", "studentVle", "vle"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquire(root, expected_sha=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "oulad.zip"
    if not archive.exists():
        temp = archive.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=120) as response, temp.open("wb") as out:
            shutil.copyfileobj(response, out)
        temp.replace(archive)
    digest = sha256(archive)
    lock = root / "manifest.json"
    previous = json.loads(lock.read_text()) if lock.exists() else None
    expected = expected_sha or (previous["archive_sha256"] if previous else EXPECTED_SHA256)
    if expected and digest != expected:
        raise ValueError("Archive SHA-256 mismatch; refusing extraction")
    with zipfile.ZipFile(archive) as z:
        # Extract only exact expected basenames. Never trust archive paths.
        for name in FILES:
            matches = [n for n in z.namelist() if Path(n).name == name + ".csv"]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name}.csv")
            with z.open(matches[0]) as source, (root / (name + ".csv")).open("wb") as out:
                shutil.copyfileobj(source, out)
    manifest = {"url": URL, "archive_sha256": digest,
                "acquired_utc": datetime.now(timezone.utc).isoformat(),
                "files": {n + ".csv": sha256(root / (n + ".csv")) for n in FILES}}
    lock.write_text(json.dumps(manifest, indent=2))
    return manifest


def load(root):
    root = Path(root)
    return {n: pd.read_csv(root / (n + ".csv"), na_values=["?"]) for n in FILES if n != "studentVle"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def foreign(child, parent, keys, label):
    require(not child[keys].isna().any().any(), f"Null foreign key: {label}")
    joined = child[keys].drop_duplicates().merge(parent[keys].drop_duplicates(),
                                                on=keys, how="left", indicator=True)
    require(joined._merge.eq("both").all(), f"Orphan keys: {label}")


def validate(root, tables=None):
    root = Path(root)
    t = tables if tables is not None else load(root)
    schemas = {
        "courses": ["code_module", "code_presentation", "module_presentation_length"],
        "studentInfo": KEY + ["final_result", "gender", "age_band", "disability",
                               "imd_band", "studied_credits", "num_of_prev_attempts"],
        "studentRegistration": KEY + ["date_registration", "date_unregistration"],
        "assessments": ["id_assessment", "code_module", "code_presentation", "assessment_type"],
        "studentAssessment": ["id_student", "id_assessment", "date_submitted", "score", "is_banked"],
        "vle": ["id_site", "code_module", "code_presentation"],
    }
    unique = {"courses": KEY[:2], "studentInfo": KEY, "studentRegistration": KEY,
              "assessments": ["id_assessment"], "studentAssessment": ["id_student", "id_assessment"],
              "vle": ["code_module", "code_presentation", "id_site"]}
    report = {}
    for name, columns in schemas.items():
        require(set(columns) <= set(t[name]), f"Missing columns in {name}")
        require(not t[name].empty, f"Empty table {name}")
        require(not t[name][unique[name]].isna().any().any(), f"Null key in {name}")
        require(not t[name].duplicated(unique[name]).any(), f"Duplicate key in {name}")
        report[name] = {"rows": len(t[name]), "missing": t[name].isna().sum().to_dict()}
    info, reg = t["studentInfo"], t["studentRegistration"]
    require(info.final_result.isin(["Pass", "Distinction", "Fail", "Withdrawn"]).all(), "Unknown outcome")
    require(t["courses"].code_presentation.str.fullmatch(r"201[34][BJ]").all(), "Unknown presentation")
    foreign(info, t["courses"], KEY[:2], "student to course")
    foreign(reg, info, KEY, "registration to student")
    foreign(info, reg, KEY, "student to registration")
    foreign(t["assessments"], t["courses"], KEY[:2], "assessment to course")
    foreign(t["studentAssessment"], t["assessments"], ["id_assessment"], "submission to assessment")
    submissions = t["studentAssessment"].merge(t["assessments"], on="id_assessment", validate="many_to_one")
    foreign(submissions, info, KEY, "submission to enrolment")
    scores = t["studentAssessment"].score.dropna()
    require(scores.between(0, 100).all(), "Score outside [0,100]")
    require(t["studentAssessment"].is_banked.isin([0, 1]).all(), "Invalid banked flag")
    require((info.studied_credits.ge(0) & info.num_of_prev_attempts.ge(0)).all(), "Negative background count")
    dates = reg.dropna(subset=["date_registration", "date_unregistration"])
    require(dates.date_registration.le(dates.date_unregistration).all(), "Withdrawal before registration")
    total = 0
    for chunk in pd.read_csv(root / "studentVle.csv", chunksize=500_000, na_values=["?"]):
        require(set(KEY + ["id_site", "date", "sum_click"]) <= set(chunk), "Missing VLE columns")
        require(chunk.date.notna().all() and chunk.sum_click.ge(0).all(), "Invalid VLE event")
        foreign(chunk, info, KEY, "VLE to enrolment")
        foreign(chunk, t["vle"], KEY[:2] + ["id_site"], "VLE to resource")
        total += len(chunk)
    require(total > 0, "Empty VLE events")
    report["studentVle"] = {"rows": total}
    return report
