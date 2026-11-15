# Feature and data contract

CSV question marks (`?`) and blank cells are parsed as missing values. This matters particularly for submission scores and registration/unregistration dates.

Join keys: `code_module`, `code_presentation`, `id_student`. Each studentInfo and studentRegistration row must be unique on these keys. Assessment IDs are globally unique; VLE resource IDs are checked together with module and presentation. VLE event rows may repeat; additive clicks are aggregated by enrolment-day before counting active days.

| Model input | Definition |
|---|---|
| code_module | Course module; one-hot encoded with unseen values ignored |
| studied_credits | Credits studied, treated as baseline numeric |
| num_of_prev_attempts | Previous attempts at the module, baseline numeric |
| date_registration | Registration day relative to presentation start; negative is valid |
| clicks | Sum of VLE sum_click from day 0 through cutoff inclusive |
| active_days | Number of distinct dates with a recorded VLE event in that window, including zero-click records |
| last_activity | Maximum recorded date in that window; -1 when no event exists |
| recent_clicks | Sum over dates greater than cutoff - 7 and no later than cutoff |

No-event counts are zero-filled; last_activity uses an explicit -1 sentinel. Other numerical missing values use training medians with indicators. Categorical missing values use the training mode. Audit missingness becomes `Unknown`; deprivation strings are nominal labels, not numeric midpoints. The target and final_result are outcome fields and never enter preprocessing.

Required raw files: courses, assessments, studentInfo, studentRegistration, studentAssessment, studentVle, vle. Assessment scores allow nulls but observed values must lie in [0,100]. VLE days can be negative or extend beyond the feature cutoff; only the feature-building window filters them. The validator reports missingness and checks invalid/null identifiers, schema omissions, duplicate primary keys, orphan relations, unknown outcomes, invalid flags, negative counts and inconsistent registration dates. It is a targeted data contract, not proof that every real-world record is correct.
