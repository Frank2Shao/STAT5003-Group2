# Run from repository root:
# Rscript run_mlr_baseline.R [path/to/acecqa_analysis.csv] [path/to/output_dir]
# Does not evaluate on test data: the shared holdout must remain untouched
# until group-wide model selection is complete.

required_packages <- c("readr", "rsample", "recipes", "parsnip",
                       "workflows", "tune", "yardstick", "glmnet")
missing_packages <- required_packages[!vapply(required_packages,
                                              requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop("Install packages first: install.packages(c(",
       paste(sprintf('"%s"', missing_packages), collapse = ", "), "))")
}

source("shared/split.R")
source("mlr/preprocessing.R")
source("mlr/model.R")
source("mlr/evaluation.R")

args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[[1]] else "data/processed/acecqa_analysis.csv"
out_dir <- if (length(args) >= 2) args[[2]] else "outputs/mlr"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- as.data.frame(readr::read_csv(data_path, show_col_types = FALSE))
stopifnot(!anyDuplicated(df$service_id))
split <- make_group_split(df, test_prop = 0.2, seed = 5003L)
save_split_manifest(df, split, file.path(out_dir, "split_manifest.csv"))
train_raw <- df[split$train, , drop = FALSE]
train <- mlr_model_data(train_raw)
folds <- make_group_folds(train_raw, v = 5L, seed = 5003L)
# Important: the resample object needs the SAME columns as the workflow.
# Rebuild each grouped split with the preselected model data.
folds_model <- lapply(folds$splits, function(s) {
  rsample::make_splits(list(analysis = s$in_id,
                            assessment = setdiff(seq_len(nrow(train)), s$in_id)), train)
})
cv <- rsample::manual_rset(folds_model, ids = folds$id)

# Audit folds, so teammates can reproduce identical provider group assignments.
fold_assign <- rep(NA_character_, nrow(train_raw))
for (i in seq_len(nrow(cv))) {
  assessment_rows <- setdiff(seq_len(nrow(train_raw)), cv$splits[[i]]$in_id)
  fold_assign[assessment_rows] <- cv$id[[i]]
}
stopifnot(!anyNA(fold_assign))
utils::write.csv(data.frame(service_id = train_raw$service_id,
                            provider_id = train_raw$provider_id,
                            fold = fold_assign),
                 file.path(out_dir, "cv_fold_manifest.csv"), row.names = FALSE)

wf <- make_mlr_workflow(train, penalty = 0.01)
set.seed(5003L)
res <- tune::fit_resamples(
  wf, resamples = cv,
  metrics = yardstick::metric_set(yardstick::accuracy, yardstick::f_meas),
  control = tune::control_resamples(save_pred = TRUE, verbose = TRUE)
)
pred <- tune::collect_predictions(res)
if (!nrow(pred)) stop("No CV predictions produced; inspect notes:",
                      paste(capture.output(tune::collect_notes(res)), collapse = "\n"))
cv_metrics <- do.call(rbind, lapply(split(pred, pred$id), function(p) {
  m <- score_mlr_predictions(p)
  m$fold <- unique(p$id)
  m
}))
utils::write.csv(cv_metrics, file.path(out_dir, "mlr_baseline_cv_metrics.csv"), row.names = FALSE)
utils::write.csv(pred, file.path(out_dir, "mlr_baseline_cv_predictions.csv"), row.names = FALSE)
print(aggregate(estimate ~ metric, cv_metrics, mean))
message("Baseline CV complete. Test set has NOT been evaluated.")
