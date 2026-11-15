# Model card

**Purpose:** research and teaching about day-28 early support in OULAD. Positive label means later failure or withdrawal among students still enrolled at the cutoff.

**Population:** historical 2013–2014 Open University module presentations. Results do not estimate performance in contemporary Nepalese, school, in-person or other university settings. The temporal test additionally excludes students seen in earlier retained splits.

**Algorithm:** validation-selected logistic regression or random forest with a constant-prevalence baseline. Exact selection, threshold, versions and metrics are recorded in reports/reference. No hyperparameter tuning or threshold selection uses test outcomes.

**Information limits:** submission date is not grade release date, so grades are excluded. Baseline credits, attempts and registration metadata are assumed available by the cutoff. Snapshot data may not reflect a production information system. Missing clicks can indicate offline study or accessibility barriers, not lack of effort.

**Errors:** false positives may cause unnecessary or stigmatizing contact; false negatives may miss students who need support. Optimize thresholds only with an agreed support capacity and harm analysis. Do not infer causal relationships from feature importance.

**Fairness:** audit four recorded attributes at a common threshold, with small-group suppression. Excluding attributes does not remove proxies. Max-minus-min gaps are descriptive and do not establish compliance or equal treatment. No intersectional guarantees are made.

**Reproducibility:** fixed archive/file hashes, fixed seed 42, pinned numerical dependencies, disjoint chronological splits, fitted preprocessing serialized with model. Numerical outputs may vary slightly by platform. The packaged model is a trusted local artifact only; joblib can execute code when loading.

**Deployment status:** not approved for operational decisions. Requires institution-specific external validation, participatory review, security/privacy governance and evidence of support benefit before any live pilot.
