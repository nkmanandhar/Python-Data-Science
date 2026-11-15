# Educational Data Science — OULAD early support

A complete Python project for retrospective early-warning research with authentic Open University Learning Analytics Dataset (OULAD) records. It downloads and verifies the original archive, validates relational tables, constructs a day-28 cohort, performs EDA, compares models, audits group errors, and produces reproducible figures and reports.

## Quick start

Use Python 3.12 (supported: 3.11–3.13). From this folder:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
oulad download
oulad validate
oulad run
```

The download is about 45 MB; extracted CSVs need about 450 MB. Allow roughly 1 GB free space plus the Python environment. Validation streams the 10-million-row VLE file; a typical laptop needs a few minutes. Network access is only needed for installation and the initial download. Commands can also use `python -m education_ds.cli` in place of `oulad`.

```bash
oulad download --data data/raw --sha256 f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4
oulad run --data data/raw --out artifacts --cutoff 28 --bootstrap 300
```

The default download already pins this hash. An upstream change fails closed. Delete a failed `.part` download and retry; a corrupt cached archive must be removed before re-downloading. Use `--sha256` to explicitly accept a separately verified release, never to bypass an unexplained mismatch. Download extraction permits only the seven known CSV basenames. Each later command verifies the CSV hashes against the local acquisition manifest.

## Question, cohort and outcome

For the exact Python 3.12 reference environment, install `requirements-lock-py312.txt` in place of `requirements.txt`. It pins transitive packages too; `requirements.txt` pins the main numerical and testing dependencies while allowing platform-compatible transitive resolution. The reference run used Python 3.12.13 on Windows.

At the end of day 28, can early engagement identify students who later fail or withdraw? The observation unit is a student–module–presentation enrolment. Target `1` is `Fail` or `Withdrawn`; target `0` is `Pass` or `Distinction`. These outcomes differ in meaning; combining them is a research simplification, not a diagnosis.

The landmark population includes students with a known registration date on or before day 28 who have not unregistered on or before that day. Unknown registration dates and earlier withdrawals are excluded and counted. Future unregistration dates affect neither model features nor prediction scores. They are consulted only to determine whether the student was still enrolled at the landmark. The pipeline retains enrolled students with no VLE events and gives them zero activity. Negative-day VLE activity is deliberately excluded; day 28 is included. `recent_clicks` covers days 22–28.

## Leakage controls

| Stage | Protocol |
|---|---|
| Training | 2013B and 2013J |
| Validation | 2014B, excluding every student in training |
| Final test | 2014J, excluding every student in training or retained validation |
| Preprocessing | Median imputation, missingness indicators, scaling and one-hot encoding fitted on training only |
| Model selection | Highest validation average precision across dummy, two logistic models and a random forest |
| Decision threshold | Validation F1 maximum over a fixed grid; deterministic tie resolution |
| Final evaluation | Fixed fitted model and threshold; no refitting on validation or test |

Different presentations have different calendars and module mixes. The test estimates temporal transfer to previously unseen students, not repeated-enrolment performance. Modules without training examples are handled by unknown-category encoding; module-specific metrics expose this shift. Enrolments of the same student may remain together within a split. Bootstrap intervals therefore resample whole students.

The explicit feature allowlist contains module, studied credits, previous attempts, registration date, and early clicks, active days, last activity and recent clicks. It excludes student IDs, presentation IDs, final results and withdrawal dates. Gender, age band, disability and deprivation band are retained solely for auditing. Education and region are omitted. Protected attributes can still be inferred from proxies; omission does not establish fairness.

Assessment tables are acquired and validated, but assessment scores are excluded from prediction. A submission timestamp does not establish when its grade became visible. OULAD has no grade-publication timestamp, so using such scores could leak future knowledge. All other historical baseline fields are assumed available at the landmark; retrospective data cannot prove their real-time availability.

## Deliverables

`artifacts/` contains:

- `validation.json`: row counts and missingness; key, foreign-key, outcome, score, date and activity checks fail loudly.
- `cohort_summary.csv`, `feature_missingness.csv`, `outcomes.png`, `activity.png`, `cohort_heatmap.png`: aggregate EDA.
- `metrics.json`: model comparison on validation, frozen test metrics, dummy baseline and 95% student-bootstrap intervals.
- `evaluation.png`: precision-recall, ROC, calibration and confusion matrix.
- `fairness.csv`, `fairness.png`: group denominators, prevalence, selection rate, precision, recall, FPR, FNR and Brier score.
- `module_metrics.csv`, `permutation_importance.csv`: transfer diagnostics and held-out AP permutation importance (descriptive, never used to reselect the model).
- `run_manifest.json`: Python/package versions, seed, source hash, feature list, cohort counts and splits.
- `model.joblib`: fitted preprocessing plus classifier, threshold, cutoff and feature contract.
- `report.md`: readable experiment report.

The executed full-data reference run is in [reports/reference/report.md](reports/reference/report.md). Its source acquisition hashes are in `reports/reference/data_manifest.json`. The trained model is supplied in the downloadable package but excluded from GitHub. Re-run the commands to regenerate it. Raw data and individual predictions are excluded from the deliverable.

## Evaluation and fairness interpretation

Average precision is the primary ranking metric; its no-skill reference is the risk prevalence in that split. ROC AUC, Brier loss, precision, recall and F1 provide complementary views. The selected threshold optimizes validation F1 for demonstration, not an agreed institutional outreach budget or harm tradeoff. The dummy baseline uses training prevalence; its fixed 0.5 decision threshold is a reference rather than a tuned outreach policy.

Fairness metrics use the same global threshold for every group. A group is suppressed if it has fewer than 30 enrolments, fewer than five positives or fewer than five negatives. Max-minus-min gaps describe only unsuppressed groups; they are not fairness certificates or hypothesis tests. Intersectional analyses and group-specific confidence intervals are not included. Bootstrap intervals quantify within-test student sampling variation, not variation across institutions, modules or presentation dates. Historic demographic categories and missing values limit interpretation. Small counts remain visible as audit denominators and should be reviewed before wider publication.

The model is intended for learning and research. It must not automatically restrict enrollment, grades or services. Any outreach use needs student participation, local validation, accessibility review, monitoring, an appeal process, and evaluation of whether support actually helps. Correlations and predictive accuracy do not demonstrate causal benefit.

## Project map

```text
src/education_ds/
  data.py       acquisition, integrity and relational validation
  features.py   landmark cohort, time-safe features and splits
  model.py      candidate models, evaluation and fairness
  report.py     EDA and diagnostic charts
  cli.py        reproducible end-to-end entry point
tests/          offline, explicitly synthetic regression fixtures
reports/reference/  executed authentic-data results
docs/           data dictionary and model card
```

Tests cover the landmark boundary, future-event invariance, future-withdrawal invariance, disjoint students, orphan rejection, exact confusion rates, small-group suppression, archive checksums/path safety, train-only imputation, unseen categories and model serialization. Fixtures are synthetic and are not used as evidence of predictive performance. The authentic full-data run separately checks integration. GitHub Actions runs offline tests on Python 3.11 and 3.12; large data downloads are not part of routine CI.

## Data provenance and attribution

OULAD by Jakub Kuzilek, Martin Hlosta and Zdenek Zdrahal is available through the [UCI dataset page](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset), DOI [10.24432/C5KK69](https://doi.org/10.24432/C5KK69). UCI licenses the data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original data descriptor is [Kuzilek et al. (2017), Scientific Data 4, 170171](https://doi.org/10.1038/sdata.2017.171). This project transforms the original records into aggregate activity features and evaluation statistics. Data authors do not endorse this project.

Project code is MIT licensed; the dataset retains its separate CC BY 4.0 terms. Do not treat the MIT license as relicensing OULAD. Never load an untrusted joblib/pickle model.
