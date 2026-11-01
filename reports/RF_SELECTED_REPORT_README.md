# RF and selected-model final-report contribution

The final report includes `_rf-report.Rmd` under Random Forest and
`_selected-model-report.Rmd` under Model Evaluation & Selection. Other members'
sections and the template placeholders remain unchanged. Keep these child files
next to `STAT5003_Final_Report.Rmd` when combining the report.

`RF_Selected_Model_Preview.Rmd` renders only Wenchen's contribution. It is a
preview, not the group's submission. Render from the repository root:

```r
rmarkdown::render("reports/RF_Selected_Model_Preview.Rmd")
rmarkdown::render("reports/STAT5003_Final_Report.Rmd")
```

Tables read the existing CSV outputs. Model scripts are included as folded,
non-executing code so rendering does not retrain models or reuse test scores for
tuning. Full modelling dependencies remain in `R/06_model_setup.R` and `mlr/`.
For original rerun instructions, see `RANDOM_FOREST_README.md` and
`SELECTED_MODEL_RESULTS_README.md`.

The report intentionally uses the corrected `outputs/mlr_shared` configuration:
lambda 0.001 and sqrt_inverse weights, not the earlier lambda 0.003 results.
Coordinate this with the MLR methods and cross-validation comparison sections.
The report does not reproduce the full five-model comparison owned by Khoa.

Before submission, the group must replace template examples, complete other
sections, consolidate references and the AI-use statement, and check the full
report against the 10-page limit. This contribution alone does not establish
that the complete report satisfies that limit.

AI assistance: OpenAI Codex assisted with drafting these sections and writing
the report-rendering and validation code. Include this assistance in the group's
AI-use disclosure; review and approve the wording and results before submission.
