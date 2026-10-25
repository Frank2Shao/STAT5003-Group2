# MLR Interpretation Add-on (train only)

Copy `mlr/interpretation.R` into the existing project's `mlr/` folder, and `run_mlr_interpretation.R` into the project root (alongside `run_mlr_tuning.R`). **Do not overwrite existing files.**

Prerequisites: existing `mlr/preprocessing.R`, `mlr/tuning.R`, processed CSV, and `outputs/mlr/split_manifest.csv`, `cv_fold_manifest.csv`, `mlr_tuning_summary.csv` produced by the earlier stages.

From project root in RStudio Console:

```r
source('run_mlr_interpretation.R')
```

Outputs in `outputs/mlr/`:

- `mlr_selected_config.csv`: CV-selected lambda and weighting scheme
- `mlr_group_comparison.csv`: one row of CV metrics formatted for group model comparison
- `mlr_selected_coefficients.csv`: fitted class-specific coefficients (including intercept)
- `mlr_selected_feature_coefficients.csv`: same, excluding intercept
- `mlr_selected_train_model.rds`: fitted train-only `glmnet` object, recipe, and configuration

**Interpreting coefficients:** This is regularized softmax multinomial regression with three class-specific coefficient vectors. `exp_coefficient` is `exp(beta)` for a class-specific linear score, **not** directly the odds ratio against a specified reference class. To compare classes A and B for a feature, use `exp(beta_A - beta_B)`; it represents the multiplicative change in the A-vs-B odds for a one-unit change of that processed feature, holding other features fixed. Numeric features have been standardized; dummy variables compare to their encoded reference levels. The selected class-weighted model optimizes weighted training loss, so coefficients should be treated as model associations, **not causal effects**. Correlated predictors and coding choices affect their interpretation.

This module **does not** inspect or evaluate held-out test rows. It uses the previously saved Provider-grouped split, all training rows, and the best CV configuration. The `.rds` is not suitable for use as a production model without further validation.

Note: the script has been checked against the previous R interfaces but R execution was not available in the authoring environment. If an error occurs, share the full RStudio console error.
