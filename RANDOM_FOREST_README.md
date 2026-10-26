# Random Forest Initial Draft

This contribution uses the existing shared setup unchanged. Keep the R/,
reports/, outputs/ and figures/ folders in the project root. It does not replace
R/06_model_setup.R or other members' models.

1. Open the project root as the working directory in RStudio.
2. Install readr, dplyr, caret, recipes, ranger, ggplot2, rmarkdown and knitr if needed.
3. Run `source("R/07_random_forest.R")` to reproduce the training and results.
4. Run `source("verify_random_forest.R")` to check output consistency.
5. Run `rmarkdown::render("reports/Random_Forest_Draft.Rmd")` for the report.

The supplied HTML can be opened directly without R. The Rmd uses saved result
files and does not retrain models when knitted. The R script overwrites only
this contribution's result files when rerun.

Git includes aggregate results and plots. Model RDS files, the split manifest,
and row-level out-of-fold predictions are generated locally by the training
script, not included in this commit. Run training before the verification script.

The report contains cross-validation results, not final held-out test results.
The saved model bundle includes its fitted recipe and selected parameters.
Load the ranger package before using predict on a model loaded from the bundle.
Any future prediction data must use the same logical-to-factor conversion as
R/06_model_setup.R, followed by recipes::bake on the saved recipe before prediction.

The shared near-zero-variance filter removes the service-type indicator. This
known issue remains intentionally unchanged for this draft. Coordinate changes
with the group and rerun models if the shared preprocessing changes.

The permutation importance plot is descriptive training/OOB output, not a
provider-grouped estimate or causal explanation. CV selected the configuration,
so these scores are subject to selection optimism. Do not tune using the test set.

AI assistance was used to draft this code and report. Review and understand the
implementation, verify the results, and disclose assistance according to the
course requirements before submission.
