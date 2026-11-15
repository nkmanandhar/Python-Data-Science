"""Run with python -m education_ds.cli download|validate|run."""
import argparse
import json
import platform
import importlib.metadata
from pathlib import Path
import joblib
import pandas as pd
from .data import acquire, load, validate, sha256, FILES, require
from .features import build, temporal_split, FEATURES
from .model import fit, metrics, bootstrap, fairness
from .report import eda, diagnostics


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "validate", "run"])
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--sha256", help="Expected archive hash for acquisition")
    parser.add_argument("--cutoff", type=int, default=28)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(acquire(args.data, args.sha256), indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    for name in FILES:
        require(sha256(args.data / (name + ".csv")) == manifest["files"][name + ".csv"],
                f"Input hash mismatch: {name}")
    tables = load(args.data)
    quality = validate(args.data, tables)
    dump(args.out / "validation.json", quality)
    if args.command == "validate":
        print("All validation checks passed")
        return
    require(args.bootstrap >= 20, "Use at least 20 bootstrap repetitions")
    d = build(args.data, tables, args.cutoff)
    train, val, test = temporal_split(d)
    model, dummy, selection = fit(train, val)
    p = model.predict_proba(test[FEATURES])[:, 1]
    threshold = selection["threshold"]
    audit, gaps = fairness(test, p, threshold)
    audit.to_csv(args.out / "fairness.csv", index=False)
    eda(d, args.out)
    diagnostics(model, test, p, threshold, audit, args.out)
    module = []
    for name, sub in test.groupby("code_module"):
        module.append({"module": name, **metrics(sub.target, p[test.code_module.eq(name)], threshold)})
    pd.DataFrame(module).to_csv(args.out / "module_metrics.csv", index=False)
    result = {"selection": selection, "test": metrics(test.target, p, threshold),
              "dummy_test": metrics(test.target, dummy.predict_proba(test[FEATURES])[:, 1], 0.5),
              "test_95pct_student_bootstrap": bootstrap(test, p, threshold, args.bootstrap),
              "fairness_max_min_gaps": gaps}
    dump(args.out / "metrics.json", result)
    run = {"python": platform.python_version(), "seed": 42, "cutoff": args.cutoff,
           "archive_sha256": manifest["archive_sha256"], "bootstrap_repetitions": args.bootstrap,
           "features": FEATURES, "raw_enrolments": len(tables["studentInfo"]), "eligible": len(d),
           "excluded_at_landmark": len(tables["studentInfo"]) - len(d),
           "student_overlap_removed": len(d) - len(train) - len(val) - len(test),
           "splits": {n: {"rows": len(s), "students": s.id_student.nunique(),
                           "presentations": sorted(s.code_presentation.unique().tolist())}
                      for n, s in [("train", train), ("validation", val), ("test", test)]},
           "versions": {p: importlib.metadata.version(p) for p in
                        ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"]}}
    dump(args.out / "run_manifest.json", run)
    joblib.dump({"pipeline": model, "threshold": threshold, "features": FEATURES,
                 "cutoff": args.cutoff}, args.out / "model.joblib")
    text = "# OULAD day-%s experiment\n\n" % args.cutoff
    text += "Source archive SHA-256: `%s`\n\n" % manifest["archive_sha256"]
    text += "Train / validation / test enrolments: %s / %s / %s.\n\n" % (len(train), len(val), len(test))
    text += "Selected model: **%s**; validation-selected threshold: %.3f.\n\n" % (selection["selected"], threshold)
    text += "| Test measure | Value |\n|---|---:|\n"
    for key in ["average_precision", "roc_auc", "brier", "precision", "recall", "f1"]:
        text += "| %s | %.4f |\n" % (key, result["test"][key])
    text += "\nSee metrics.json for student-bootstrap intervals and baseline, fairness.csv for group denominators and suppressed groups, and module_metrics.csv for module shift.\n\n"
    text += "The selected threshold flags %.1f%% of test enrolments, with %.1f%% precision and %.1f%% false-positive rate. This is a high-volume review policy, not a practical support capacity recommendation.\n\n" % (100*result["test"]["selection_rate"], 100*result["test"]["precision"], 100*result["test"]["fpr"])
    text += "The unsuppressed age-band recall gap is %.1f percentage points. Group errors differ; inspect denominators before interpreting gaps. No fairness guarantee follows from these results.\n\n" % (100*gaps["age_band"]["recall"]) if gaps["age_band"]["recall"] is not None else ""
    text += "![Evaluation](evaluation.png)\n\n![Activity](activity.png)\n\n![Fairness](fairness.png)\n\n"
    text += "These are retrospective associations on historical UK distance-learning cohorts. They do not establish that outreach improves outcomes or that the model is fair. Excluding overlapping students changes the evaluation population. Grade timestamps are unavailable, so assessment scores are excluded. No deployment is authorized by these results.\n"
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
