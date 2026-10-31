# Part 3: selected model and test results

This contribution preserves the existing MLR code and outputs. It reuses Wilson's recipe, ridge multinomial model and full tuning grid, but writes a separate run based on the actual shared split and five folds from `R/06_model_setup.R`.

## Why a separate run was needed

The original `outputs/mlr/split_manifest.csv` had 13,762 training services, rather than the shared 13,497. It included 3,194 of the 3,663 shared test services. Its CV folds also differed. The comment in `mlr/tuning.R` did not establish fold equivalence; row-level manifest checks found the mismatch. The original `run_mlr_baseline.R` generates its own split and must not be used to regenerate the corrected manifests.

The corrected run has zero provider overlap between training and test, and exact shared CV assignments. It leaves the original files unchanged.

## Results

| Candidate | Mean CV Macro-F1 | Fold SD | Mean CV accuracy |
|:---|---:|---:|---:|
| MLR, shared split | 0.4871 | 0.0180 | 72.55% |
| Random Forest | 0.4825 | 0.0265 | 63.02% |
| SVM (RBF) | 0.4696 | 0.0149 | 70.95% |
| LDA | 0.4364 | 0.0189 | 51.06% |
| KNN | 0.4246 | 0.0247 | 72.24% |

The corrected MLR remains the highest-scoring candidate under the planned criterion. The gap to RF is only 0.00465; no statistically significant advantage is claimed. These compare model pipelines, not algorithms with identical preprocessing: each retains its original preparation and features.

Selected ridge penalty: 0.001 (previous run: 0.003). Weighting: square-root inverse frequency. The choice was recorded before scoring the test partition.

On 3,663 test services: Macro-F1 0.4542, accuracy 73.25%. The majority baseline scores 0.2885 and 76.28%. Recall is 8.1% Below, 86.5% Meeting, 39.5% Above. Only 20 of 248 Below NQS services are identified.

## Important limitation

This corrected fit does not train or tune on the shared test rows. However, the original MLR training and LDA evaluation already exposed this partition during the project. Full-data EDA also preceded modelling. Refitting cannot erase that history. Report the test results as a corrected-fit evaluation on the existing partition, not as a pristine independent final confirmation. No retuning or switching models based on the test results was done here. A fresh external dataset would be needed for stronger independent confirmation.

## Group integration

- Khoa: use `outputs/mlr_shared/` instead of `outputs/mlr/` for the three MLR inputs in the comparison QMD. The corrected all-model summary is in `outputs/selected_model/comparison_summary.csv`.
- Wilson: the selected penalty and coefficients changed after alignment. For interpretation of this refit, use the new `outputs/mlr_shared/mlr_selected_feature_coefficients.csv` and saved model, not the original coefficients. The original method remains credited to you.
- Add `{{< include _selected-model-results.qmd >}}` to the main deck in `presentation/`. Keep the figure in its existing relative path. This adds one slide with English speaker notes.
- The RF slides are a separate PR (#19); this PR does not replace them. This is a proposed integration, not a claim that the team has already accepted the correction.

## Reproduce

From the repository root, with the shared-workflow packages plus `glmnet` installed:

```r
source("R/11_mlr_shared_split.R")
source("R/12_selected_model_test.R")
source("verify_selected_model.R")
```

Stage 11 runs 18 MLR configurations over five folds and refits the winner on training only. Stage 12 compares saved CV scores, requires MLR to remain the top candidate, audits preprocessing and manifests, then scores the fixed model. Re-running stage 12 reproduces the same fixed evaluation, not a new independent experiment. Never use it to choose parameters after seeing test performance.

```sh
quarto render presentation/selected_model_results.qmd
```

Rendering uses saved evidence and does not retrain or evaluate models. The generated preview is local; source QMD and evidence are versioned. The large corrected tuning OOF file is reproducible but excluded from Git. The compact fitted MLR is included for teammates to inspect and use without retraining.

## Reference provenance

The three files under `outputs/selected_model/reference/` are unchanged CSV snapshots from Khoa's `lda-presentation` commit `ef99b35c4e4f0a80fffc76ddbd0d2efac149f749`: `outputs/lda/lda_cv_fold_metrics.csv`, `outputs/knn/cv_fold_metrics.csv`, and `outputs/knn/best_parameters.csv`. They support reproducible comparison without merging teammates' unfinished branches. RF and SVM scores are from the existing main-branch outputs. Khoa's aggregation approach is retained: unweighted means across the five folds. Other models were not retrained here.

AI assistance was used for the audit, rerun orchestration, result checks, plot, slide text and speaker notes. The group should review these outputs and include that assistance in its disclosure. The group's SOLES template and existing attribution are retained.
