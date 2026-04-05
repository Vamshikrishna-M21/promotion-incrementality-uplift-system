# Promotion Incrementality and Uplift Decision System

This project builds an end-to-end causal targeting pipeline for promotions using randomized treatment/control data. Instead of predicting who is likely to convert, the system estimates who is likely to convert **because of treatment**, then turns those estimates into budget-aware targeting decisions.

The project covers the full decision workflow:

- estimate average treatment effect from a randomized experiment
- model heterogeneous treatment effects with uplift methods
- rank users by predicted incremental impact
- evaluate targeting policies under budget and ROI constraints
- compare uplift targeting against naive business baselines

## Dataset

This project uses the **Criteo Uplift Prediction Dataset**, a public randomized treatment/control benchmark for uplift modeling and promotion targeting.

Why this dataset is a strong fit:

- It supports clean average treatment effect estimation because treatment assignment is randomized.
- It is built for uplift modeling, so the core task is causal targeting rather than standard response prediction.
- It is large enough to support meaningful model comparison and policy evaluation.
- It matches the business question directly: which users should receive treatment to maximize incremental value?

Practical considerations:

- The features are anonymized, so the project emphasizes decision quality and causal evaluation more than feature interpretation.
- The public dataset is large, so the pipeline creates a reproducible random sample for faster local experimentation.
- The data does not have panel or time-series structure, so methods like Difference-in-Differences, Synthetic Control, and SDID are intentionally excluded.

## Methodology

The pipeline is organized in the same order a production decisioning workflow would be built.

1. **Randomized experiment analysis**
   Estimate average treatment effects with difference-in-means, confidence intervals, and significance tests.
2. **Uplift / HTE modeling**
   Train user-level treatment effect models with two-model logistic regression, two-model gradient boosting, and interaction logistic regression.
3. **Model selection**
   Choose the best uplift model on a validation split using uplift curves, AUUC, and Qini.
4. **Policy evaluation**
   Convert uplift scores into budgeted targeting policies and estimate offline policy value with inverse-propensity weighting.
5. **Business comparison**
   Compare uplift targeting against naive response targeting, random targeting, treat-all, and treat-none baselines.

## Repo Layout

```text
configs/
  default.yaml
src/promo_uplift/
  ab_testing.py
  cli.py
  config.py
  data.py
  models.py
  pipeline.py
  plots.py
  policy.py
tests/
reports/
  figures/
  metrics/
  summary.md
```

## Reproducing the Project

```bash
make setup
make test
make run
```

The default run downloads the public Criteo file, builds a random sample of about 300k rows, trains the candidate uplift models, and writes plots and metrics under `reports/`.

## Main Results From the Current Run

Run summary:

- Random sample size: **299,421**
- Treatment propensity in the test split: **0.8481**
- Best uplift model by validation AUUC: **two_model_logistic**

Holdout experiment results:

- **Conversion ATE:** `+0.001512` absolute lift, 95% CI `[0.000575, 0.002450]`, `p=0.00157`
- **Visit ATE:** `+0.012007` absolute lift, 95% CI `[0.007783, 0.016231]`, `p=2.52e-08`

Validation uplift ranking:

- `two_model_logistic`: `0.001250`
- `interaction_logistic`: `0.001002`
- `two_model_gbdt`: `0.000990`

Targeting takeaways with `conversion_value=100` and `treatment_cost=1`:

- Uplift targeting beats random and treat-all at tight budgets.
- At a **5% budget**, uplift targeting produces positive incremental value per eligible user (`0.0445`), slightly ahead of naive response targeting (`0.0414`).
- As the budget expands, the marginal population becomes less incremental and policy value turns negative.
- Treating everyone is strongly dominated under the current cost/value assumptions.

## Visuals

### Randomized Lift

![ATE summary](reports/figures/ate_summary.png)

### Uplift Curves

![Uplift curves](reports/figures/uplift_curves.png)

### Budgeted Policy Value

![Policy value](reports/figures/policy_value.png)

## Notes on Scope

- This is a **causal decision system**, not a plain conversion predictor.
- The `exposure` column is intentionally not used as a modeling target because it is post-treatment.
- DiD, Synthetic Control, and SDID are intentionally excluded because the selected data does not support panel or temporal identification.

## Tests

```bash
make test
```

The tests cover core causal effect and policy-evaluation utilities so the project is not just a report-generating script.
