import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from education_ds.data import acquire, foreign, sha256, load, validate
from education_ds.features import build, temporal_split, FEATURES, AUDIT
from education_ds.model import metrics, fairness, fit, bootstrap


@pytest.fixture
def source(tmp_path):
    key = {"code_module": "AAA", "code_presentation": "2013B"}
    info = pd.DataFrame([{**key, "id_student": i, "final_result": "Fail" if i % 2 else "Pass",
                          "gender": "F", "age_band": "0-35", "disability": "N", "imd_band": "0-10%",
                          "studied_credits": 60, "num_of_prev_attempts": 0} for i in range(1, 5)])
    reg = pd.DataFrame([{**key, "id_student": i, "date_registration": -10 if i != 4 else 29,
                         "date_unregistration": 28 if i == 3 else np.nan} for i in range(1, 5)])
    events = pd.DataFrame([{**key, "id_student": 1, "id_site": 1, "date": day, "sum_click": clicks}
                           for day, clicks in [(-1, 100), (0, 2), (22, 3), (28, 5), (29, 999)]])
    events.to_csv(tmp_path / "studentVle.csv", index=False)
    return tmp_path, {"studentInfo": info, "studentRegistration": reg}


def test_landmark_and_inclusive_cutoff(source):
    root, tables = source
    d = build(root, tables)
    assert d.id_student.tolist() == [1, 2]
    assert d.clicks.tolist() == [10, 0]
    assert d.active_days.tolist() == [3, 0]
    assert d.recent_clicks.tolist() == [8, 0]
    assert d.last_activity.tolist() == [28, -1]
    assert d.target.tolist() == [1, 0]


def test_future_events_do_not_change_features(source):
    root, tables = source
    before = build(root, tables)
    events = pd.read_csv(root / "studentVle.csv")
    events.loc[events.date.gt(28), "sum_click"] = 10_000_000
    events.to_csv(root / "studentVle.csv", index=False)
    pd.testing.assert_frame_equal(before, build(root, tables))


def test_future_withdrawal_not_a_feature(source):
    root, tables = source
    before = build(root, tables)
    tables["studentRegistration"].loc[0, "date_unregistration"] = 99
    after = build(root, tables)
    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])
    assert not set(FEATURES) & set(AUDIT + ["id_student", "target", "final_result", "date_unregistration"])


def test_foreign_keys_fail_loudly():
    with pytest.raises(ValueError, match="Orphan"):
        foreign(pd.DataFrame({"id": [2]}), pd.DataFrame({"id": [1]}), ["id"], "test")


def test_split_purges_all_student_overlap():
    d = pd.DataFrame({"code_presentation": ["2013B"] * 2 + ["2014B"] * 3 + ["2014J"] * 4,
                      "id_student": [1, 2, 1, 3, 4, 1, 3, 5, 6],
                      "target": [0, 1, 0, 0, 1, 0, 0, 0, 1]})
    train, val, test = temporal_split(d)
    assert set(train.id_student) == {1, 2}
    assert set(val.id_student) == {3, 4}
    assert set(test.id_student) == {5, 6}


def test_metrics_and_zero_denominators():
    m = metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
    assert m["recall"] == m["fpr"] == m["precision"] == 0.5
    assert metrics([0, 0], [0.1, 0.2], 0.5)["fnr"] is None


def test_fairness_suppresses_tiny_groups():
    d = pd.DataFrame({attr: ["a", "b"] for attr in AUDIT})
    d["target"] = [0, 1]
    audit, gaps = fairness(d, [0.1, 0.9], 0.5)
    assert audit.suppressed.all()
    assert gaps["gender"]["recall"] is None


def test_acquisition_hash_rejection_and_safe_paths(tmp_path, monkeypatch):
    from education_ds import data
    with zipfile.ZipFile(tmp_path / "oulad.zip", "w") as z:
        for name in data.FILES:
            z.writestr("../../" + name + ".csv", "id\n1\n")
        z.writestr("../../escape.txt", "bad")
    with pytest.raises(ValueError, match="SHA-256"):
        acquire(tmp_path, "0" * 64)
    manifest = acquire(tmp_path, sha256(tmp_path / "oulad.zip"))
    assert len(manifest["files"]) == 7
    assert not (tmp_path.parent / "escape.txt").exists()


def test_training_preprocessor_and_artifact_roundtrip(tmp_path):
    import joblib
    rng = np.random.default_rng(42)
    n = 80
    d = pd.DataFrame({name: rng.normal(size=n) for name in FEATURES if name != "code_module"})
    d["code_module"] = ["AAA"] * n
    d["target"] = (d.clicks > 0).astype(int)
    train, val = d.iloc[:50].copy(), d.iloc[50:].copy()
    val["code_module"] = "UNSEEN"
    model, dummy, selection = fit(train, val)
    transformer = model.named_steps["preprocess"].named_transformers_["numeric"]
    assert transformer.named_steps["impute"].statistics_[0] == pytest.approx(train.studied_credits.median())
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(model.predict_proba(val[FEATURES]), joblib.load(path).predict_proba(val[FEATURES]))
    assert 0.05 <= selection["threshold"] <= 0.95


def test_authentic_missing_marker_and_validation(source):
    root, t = source
    key = {"code_module": "AAA", "code_presentation": "2013B"}
    t["courses"] = pd.DataFrame([{**key, "module_presentation_length": 240}])
    t["assessments"] = pd.DataFrame([{**key, "id_assessment": 1, "assessment_type": "TMA"}])
    t["studentAssessment"] = pd.DataFrame([{"id_student": 1, "id_assessment": 1,
                                             "date_submitted": 14, "score": np.nan, "is_banked": 0}])
    t["vle"] = pd.DataFrame([{**key, "id_site": 1}])
    for name, table in t.items():
        table.to_csv(root / (name + ".csv"), index=False, na_rep="?")
    loaded = load(root)
    assert loaded["studentAssessment"].score.isna().all()
    assert loaded["studentRegistration"].date_unregistration.isna().sum() == 3
    report = validate(root, loaded)
    assert report["studentVle"]["rows"] == 5
    loaded["studentInfo"] = pd.concat([loaded["studentInfo"], loaded["studentInfo"].iloc[:1]])
    with pytest.raises(ValueError, match="Duplicate key"):
        validate(root, loaded)


def test_student_bootstrap_reproducible():
    test = pd.DataFrame({"id_student": [1, 1, 2, 2, 3, 4], "target": [0, 1, 0, 1, 0, 1]})
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    a = bootstrap(test, p, 0.5, repetitions=20)
    assert a == bootstrap(test, p, 0.5, repetitions=20)
    assert a["roc_auc"] == [1.0, 1.0]
