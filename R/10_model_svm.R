# -------------------------------------------------------------------
# Linear Support Vector Machine
# -------------------------------------------------------------------

required_svm_packages <- c(
  "kernlab"
)

missing_svm_packages <- required_svm_packages[
  !vapply(
    required_svm_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_svm_packages) > 0) {
  stop(
    "Install the following packages before running this script: ",
    paste(missing_svm_packages, collapse = ", ")
  )
}

library(kernlab)

source("R/06_model_setup.R")

# -------------------------------------------------------------------
# Phase 0: input and resampling checks
# -------------------------------------------------------------------

# this section aims to confirm that the shared workflow created the expected
# outcome,predictor set and five-fold resampling structure. Identifier leakage
# and provider overlap must be absent before model fitting begins.

expected_outcome_levels <- c(
  "Below NQS",
  "Meeting NQS",
  "Above NQS"
)

forbidden_predictors <- c(
  "service_id",
  "provider_id",
  "overall_rating_original"
)

split_provider_overlap <- length(
  intersect(
    unique(training_data$provider_id),
    unique(test_data$provider_id)
  )
)

fold_provider_overlap <- vapply(
  names(cv_index),
  function(fold_name) {
    analysis_providers <- unique(
      training_data$provider_id[cv_index[[fold_name]]]
    )

    validation_providers <- unique(
      training_data$provider_id[cv_index_out[[fold_name]]]
    )

    length(
      intersect(
        analysis_providers,
        validation_providers
      )
    )
  },
  integer(1)
)

stopifnot(
  identical(
    levels(training_model_data$rating_tier),
    expected_outcome_levels
  ),
  !any(forbidden_predictors %in% names(training_model_data)),
  identical(
    names(training_model_data),
    names(test_model_data)
  ),
  length(cv_index) == 5L,
  length(cv_index_out) == 5L,
  identical(names(cv_index), names(cv_index_out)),
  split_provider_overlap == 0L,
  all(fold_provider_overlap == 0L)
)

cat(
  "\nPhase 0 checks passed",
  "\nTraining dimensions:",
  nrow(training_model_data), "rows x",
  ncol(training_model_data), "columns",
  "\nTest dimensions:",
  nrow(test_model_data), "rows x",
  ncol(test_model_data), "columns",
  "\nNumber of predictors:",
  ncol(training_model_data) - 1L,
  "\nNumber of CV folds:",
  length(cv_index),
  "\nTrain-test provider overlap:",
  split_provider_overlap,
  "\nMaximum within-fold provider overlap:",
  max(fold_provider_overlap),
  "\n\nTraining outcome counts:\n"
)

print(
  table(training_model_data$rating_tier)
)


# -------------------------------------------------------------------
# Phase 1: Linear SVM single-parameter pilot
# -------------------------------------------------------------------

# Fit C = 1 first to validate the complete recipe-model-metric pipeline before
# launching a larger tuning grid. Five-fold CV means that this single candidate
# requires five separate model fits.

svm_linear_pilot_grid <- data.frame(
  C = 1
)

set.seed(5005L)

svm_linear_pilot_time <- system.time({
  svm_linear_pilot <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = model_control,
    tuneGrid = svm_linear_pilot_grid,
    metric = "Macro_F1",
    maximize = TRUE
  )
})

cat(
  "\nLinear SVM pilot completed.\n",
  "Candidate C:", svm_linear_pilot_grid$C, "\n"
)

print(svm_linear_pilot)
print(svm_linear_pilot_time)

# -------------------------------------------------------------------
# Phase 1: pilot validation
# -------------------------------------------------------------------

# Check that all five resamples completed and that every training observation
# received exactly one out-of-fold prediction. These predictions provide
# internal validation without using the test set.

pilot_metric_names <- c(
  "Macro_F1",
  "Accuracy",
  "Recall_Below.NQS",
  "Recall_Meeting.NQS",
  "Recall_Above.NQS"
)

pilot_predictions <- svm_linear_pilot$pred
pilot_resamples <- svm_linear_pilot$resample

predictions_per_row <- table(
  pilot_predictions$rowIndex
)

stopifnot(
  nrow(pilot_resamples) == 5L,
  all(pilot_metric_names %in% names(pilot_resamples)),
  !anyNA(pilot_resamples[, pilot_metric_names]),
  nrow(pilot_predictions) == nrow(training_model_data),
  setequal(
    pilot_predictions$rowIndex,
    seq_len(nrow(training_model_data))
  ),
  all(predictions_per_row == 1L),
  !anyNA(
    pilot_predictions[
      ,
      c("pred", "obs", "rowIndex", "Resample", "C")
    ]
  ),
  all(pilot_predictions$C == 1)
)

cat(
  "\nPilot validation passed.",
  "\nCompleted resamples:", nrow(pilot_resamples),
  "\nSaved out-of-fold predictions:", nrow(pilot_predictions),
  "\nMissing predictions: 0",
  "\n\nFold-level metrics:\n"
)

print(
  pilot_resamples[
    ,
    c("Resample", pilot_metric_names)
  ]
)

cat("\nOut-of-fold predicted class counts:\n")

print(
  table(pilot_predictions$pred)
)

# Pilot interpretation:
# At C = 1, all five folds predicted every observation as Meeting NQS.
# Macro-F1 therefore remained near the majority-class baseline, while recall
# for Below NQS and Above NQS was zero. Coarse C tuning is required before
# class weighting should be applied.


# -------------------------------------------------------------------
# Phase 2: Linear SVM coarse tuning
# -------------------------------------------------------------------

# First time I tried 0.03125, 0.125, 0.5, 2 and 8, but it took more
# than 8 hours to process, so change plan to test candidates separately so that
# runtime and model behaviour can be inspected before moving to a larger C.
# Fold-level progress is displayed to identify where an unexpectedly long fit
# occurs.

coarse_control_verbose <- model_control
coarse_control_verbose$verboseIter <- TRUE

# First candidate: C = 2^(-3) = 0.125

set.seed(5007L)

svm_linear_C_0125_time <- system.time({
  svm_linear_C_0125 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(C = 2^(-3)),
    metric = "Macro_F1",
    maximize = TRUE
  )
})

cat(
  "\nLinear SVM candidate completed.",
  "\nC:", 2^(-3),
  "\n\nCross-validation results:\n"
)

print(svm_linear_C_0125$results)

cat("\nElapsed time:\n")
print(svm_linear_C_0125_time)

# Second candidate: C = 2^(-1) = 0.5

set.seed(5008L)

svm_linear_C_05_time <- system.time({
  svm_linear_C_05 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(C = 2^(-1)),
    metric = "Macro_F1",
    maximize = TRUE
  )
})

cat(
  "\nLinear SVM candidate completed.",
  "\nC:", 2^(-1),
  "\n\nCross-validation results:\n"
)

print(svm_linear_C_05$results)

cat("\nElapsed time:\n")
print(svm_linear_C_05_time)

# Third candidate: C = 2^(1) = 2

set.seed(5009L)

svm_linear_C_2_time <- system.time({
  svm_linear_C_2 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(C = 2^1),
    metric = "Macro_F1",
    maximize = TRUE
  )
})

cat(
  "\nLinear SVM candidate completed.",
  "\nC:", 2^1,
  "\n\nCross-validation results:\n"
)

print(svm_linear_C_2$results)

cat("\nElapsed time:\n")
print(svm_linear_C_2_time)

# Lower-bound candidate: C = 2^(-5) = 0.03125

set.seed(5010L)

svm_linear_C_003125_time <- system.time({
  svm_linear_C_003125 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(C = 2^(-5)),
    metric = "Macro_F1",
    maximize = TRUE
  )
})

cat(
  "\nLinear SVM candidate completed.",
  "\nC:", 2^(-5),
  "\n\nCross-validation results:\n"
)

print(svm_linear_C_003125$results)

cat("\nElapsed time:\n")
print(svm_linear_C_003125_time)

# Extended lower-bound candidate: C = 2^(-7) = 0.0078125

set.seed(5011L)

svm_linear_C_00078125_time <- system.time({
  svm_linear_C_00078125 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(C = 2^(-7)),
    metric = "Macro_F1",
    maximize = TRUE
  )
})

cat(
  "\nLinear SVM candidate completed.",
  "\nC:", 2^(-7),
  "\n\nCross-validation results:\n"
)

print(svm_linear_C_00078125$results)

cat("\nElapsed time:\n")
print(svm_linear_C_00078125_time)

# -------------------------------------------------------------------
# Phase 2: unweighted tuning summary
# -------------------------------------------------------------------

# Combine the separately fitted candidates into one comparison table.
# The pilot at C = 1 is documented separately; it produced the same
# majority-class collapse as every candidate listed below.

svm_result_columns <- c(
  "C",
  "Macro_F1",
  "Macro_F1SD",
  "Accuracy",
  "AccuracySD",
  "Recall_Below.NQS",
  "Recall_Meeting.NQS",
  "Recall_Above.NQS"
)

svm_linear_unweighted_results <- rbind(
  svm_linear_C_00078125$results[, svm_result_columns],
  svm_linear_C_003125$results[, svm_result_columns],
  svm_linear_C_0125$results[, svm_result_columns],
  svm_linear_C_05$results[, svm_result_columns],
  svm_linear_C_2$results[, svm_result_columns]
)

svm_linear_unweighted_results <-
  svm_linear_unweighted_results[
    order(svm_linear_unweighted_results$C),
  ]

rownames(svm_linear_unweighted_results) <- NULL

unweighted_reference_C <- 2^(-5)

unweighted_reference_result <-
  svm_linear_unweighted_results[
    svm_linear_unweighted_results$C == unweighted_reference_C,
  ]

stopifnot(
  nrow(unweighted_reference_result) == 1L,
  all(svm_linear_unweighted_results$Recall_Below.NQS == 0),
  all(svm_linear_unweighted_results$Recall_Meeting.NQS == 1),
  all(svm_linear_unweighted_results$Recall_Above.NQS == 0)
)

cat(
  "\nUnweighted Linear SVM tuning summary:\n"
)

print(svm_linear_unweighted_results)

cat(
  "\nUnweighted reference C:", unweighted_reference_C,
  "\nReason: all tested candidates produced identical CV metrics;",
  "\nC = 0.03125 is the smallest value in the planned coarse grid.",
  "\nLarger values were increasingly expensive, so C = 8 and C = 32",
  "\nwere not retained for further unweighted evaluation.\n"
)


# -------------------------------------------------------------------
# Phase 3: class-weighted Linear SVM
# -------------------------------------------------------------------

# Macro-F1 changes model evaluation but does not change the fitting
# objective. Square-root inverse-frequency weights moderately increase
# the penalty for errors in under-represented outcome classes.

training_class_counts <- table(
  training_model_data$rating_tier
)

svm_class_weights <- sqrt(
  max(training_class_counts) / training_class_counts
)

svm_class_weights <- as.numeric(svm_class_weights)

names(svm_class_weights) <- names(training_class_counts)

stopifnot(
  identical(
    names(svm_class_weights),
    levels(training_model_data$rating_tier)
  ),
  svm_class_weights["Meeting NQS"] == 1,
  all(svm_class_weights > 0)
)

svm_class_weight_table <- data.frame(
  rating_tier = names(training_class_counts),
  n_training = as.integer(training_class_counts),
  class_weight = as.numeric(svm_class_weights),
  row.names = NULL
)

cat(
  "\nSquare-root inverse-frequency class weights:\n"
)

print(svm_class_weight_table)

# Weighted candidate at the unweighted reference C

set.seed(5012L)

svm_linear_weighted_C_003125_time <- system.time({
  svm_linear_weighted_C_003125 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(C = unweighted_reference_C),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nClass-weighted Linear SVM completed.",
  "\nC:", unweighted_reference_C,
  "\n\nCross-validation results:\n"
)

print(svm_linear_weighted_C_003125$results)

cat("\nElapsed time:\n")
print(svm_linear_weighted_C_003125_time)

# Weighted coarse candidate: C = 2^(-7) = 0.0078125

set.seed(5013L)

svm_linear_weighted_C_00078125_time <- system.time({
  svm_linear_weighted_C_00078125 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(C = 2^(-7)),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nClass-weighted Linear SVM completed.",
  "\nC:", 2^(-7),
  "\n\nCross-validation results:\n"
)

print(svm_linear_weighted_C_00078125$results)

cat("\nElapsed time:\n")
print(svm_linear_weighted_C_00078125_time)

# Weighted coarse candidate: C = 2^(-3) = 0.125

set.seed(5014L)

svm_linear_weighted_C_0125_time <- system.time({
  svm_linear_weighted_C_0125 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(C = 2^(-3)),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nClass-weighted Linear SVM completed.",
  "\nC:", 2^(-3),
  "\n\nCross-validation results:\n"
)

print(svm_linear_weighted_C_0125$results)

cat("\nElapsed time:\n")
print(svm_linear_weighted_C_0125_time)

# Weighted coarse candidate: C = 2^(-1) = 0.5

set.seed(5015L)

svm_linear_weighted_C_05_time <- system.time({
  svm_linear_weighted_C_05 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(C = 2^(-1)),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nClass-weighted Linear SVM completed.",
  "\nC:", 2^(-1),
  "\n\nCross-validation results:\n"
)

print(svm_linear_weighted_C_05$results)

cat("\nElapsed time:\n")
print(svm_linear_weighted_C_05_time)

# Weighted fine-tuning candidate: C = 0.0625

set.seed(5016L)

svm_linear_weighted_C_00625_time <- system.time({
  svm_linear_weighted_C_00625 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(C = 0.0625),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nClass-weighted Linear SVM completed.",
  "\nC:", 0.0625,
  "\n\nCross-validation results:\n"
)

print(svm_linear_weighted_C_00625$results)

cat("\nElapsed time:\n")
print(svm_linear_weighted_C_00625_time)

# Weighted fine-tuning candidate: C = 0.25

set.seed(5017L)

svm_linear_weighted_C_025_time <- system.time({
  svm_linear_weighted_C_025 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmLinear",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(C = 0.25),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nClass-weighted Linear SVM completed.",
  "\nC:", 0.25,
  "\n\nCross-validation results:\n"
)

print(svm_linear_weighted_C_025$results)

cat("\nElapsed time:\n")
print(svm_linear_weighted_C_025_time)

# -------------------------------------------------------------------
# Phase 3: weighted tuning summary
# -------------------------------------------------------------------

# Compare weighted candidates using both their mean CV metrics and
# paired fold-level Macro-F1 differences. The paired comparison uses
# identical validation folds for every candidate.

svm_linear_weighted_results <- rbind(
  svm_linear_weighted_C_00078125$results[, svm_result_columns],
  svm_linear_weighted_C_003125$results[, svm_result_columns],
  svm_linear_weighted_C_00625$results[, svm_result_columns],
  svm_linear_weighted_C_0125$results[, svm_result_columns],
  svm_linear_weighted_C_025$results[, svm_result_columns],
  svm_linear_weighted_C_05$results[, svm_result_columns]
)

svm_linear_weighted_results <-
  svm_linear_weighted_results[
    order(svm_linear_weighted_results$C),
  ]

rownames(svm_linear_weighted_results) <- NULL

weighted_resample_results <- rbind(
  data.frame(
    C = 0.0078125,
    svm_linear_weighted_C_00078125$resample
  ),
  data.frame(
    C = 0.03125,
    svm_linear_weighted_C_003125$resample
  ),
  data.frame(
    C = 0.0625,
    svm_linear_weighted_C_00625$resample
  ),
  data.frame(
    C = 0.125,
    svm_linear_weighted_C_0125$resample
  ),
  data.frame(
    C = 0.25,
    svm_linear_weighted_C_025$resample
  ),
  data.frame(
    C = 0.5,
    svm_linear_weighted_C_05$resample
  )
)

stopifnot(
  !anyNA(weighted_resample_results$Macro_F1),
  all(
    table(weighted_resample_results$C) ==
      length(cv_index)
  )
)

weighted_fold_macro_f1 <- xtabs(
  Macro_F1 ~ Resample + C,
  data = weighted_resample_results
)

highest_mean_C <- svm_linear_weighted_results$C[
  which.max(svm_linear_weighted_results$Macro_F1)
]

reference_fold_results <- weighted_resample_results[
  weighted_resample_results$C == highest_mean_C,
  c("Resample", "Macro_F1")
]

names(reference_fold_results)[2] <- "Reference_Macro_F1"

paired_fold_differences <- merge(
  weighted_resample_results[
    ,
    c("C", "Resample", "Macro_F1")
  ],
  reference_fold_results,
  by = "Resample"
)

paired_fold_differences$difference_from_highest_mean <-
  paired_fold_differences$Macro_F1 -
  paired_fold_differences$Reference_Macro_F1

paired_difference_summary <- do.call(
  rbind,
  lapply(
    split(
      paired_fold_differences,
      paired_fold_differences$C
    ),
    function(candidate_data) {
      data.frame(
        C = unique(candidate_data$C),
        mean_difference = mean(
          candidate_data$difference_from_highest_mean
        ),
        minimum_difference = min(
          candidate_data$difference_from_highest_mean
        ),
        maximum_difference = max(
          candidate_data$difference_from_highest_mean
        )
      )
    }
  )
)

rownames(paired_difference_summary) <- NULL

cat("\nWeighted Linear SVM tuning results:\n")
print(svm_linear_weighted_results)

cat("\nFold-level Macro-F1 by C:\n")
print(weighted_fold_macro_f1)

cat(
  "\nCandidate with highest mean Macro-F1:",
  highest_mean_C,
  "\n\nPaired differences relative to that candidate:\n"
)

print(
  paired_difference_summary[
    order(paired_difference_summary$C),
  ]
)

# -------------------------------------------------------------------
# Phase 4: selected-model out-of-fold diagnostics
# -------------------------------------------------------------------

# C = 0.125 is retained because its performance is practically
# indistinguishable from C = 0.5 while applying stronger regularisation.
# All diagnostics below use predictions from held-out CV folds only.

selected_weighted_C <- 0.125

svm_linear_selected <- svm_linear_weighted_C_0125

selected_oof_predictions <- svm_linear_selected$pred[
  svm_linear_selected$pred$C == selected_weighted_C,
]

selected_class_levels <- levels(
  training_model_data$rating_tier
)

selected_oof_predictions$obs <- factor(
  selected_oof_predictions$obs,
  levels = selected_class_levels
)

selected_oof_predictions$pred <- factor(
  selected_oof_predictions$pred,
  levels = selected_class_levels
)

predictions_per_training_row <- table(
  selected_oof_predictions$rowIndex
)

stopifnot(
  nrow(selected_oof_predictions) ==
    nrow(training_model_data),
  setequal(
    selected_oof_predictions$rowIndex,
    seq_len(nrow(training_model_data))
  ),
  all(predictions_per_training_row == 1L),
  !anyNA(
    selected_oof_predictions[
      ,
      c("obs", "pred", "rowIndex", "Resample")
    ]
  )
)

svm_selected_confusion_matrix <- table(
  Observed = selected_oof_predictions$obs,
  Predicted = selected_oof_predictions$pred
)

true_positive <- diag(
  svm_selected_confusion_matrix
)

predicted_total <- colSums(
  svm_selected_confusion_matrix
)

observed_total <- rowSums(
  svm_selected_confusion_matrix
)

class_precision <- ifelse(
  predicted_total == 0,
  0,
  true_positive / predicted_total
)

class_recall <- ifelse(
  observed_total == 0,
  0,
  true_positive / observed_total
)

class_f1 <- ifelse(
  class_precision + class_recall == 0,
  0,
  2 * class_precision * class_recall /
    (class_precision + class_recall)
)

svm_selected_class_metrics <- data.frame(
  rating_tier = selected_class_levels,
  support = as.integer(observed_total),
  precision = as.numeric(class_precision),
  recall = as.numeric(class_recall),
  F1 = as.numeric(class_f1),
  row.names = NULL
)

svm_selected_pooled_metrics <- data.frame(
  metric = c(
    "Pooled_OOF_Macro_F1",
    "Pooled_OOF_Accuracy",
    "Mean_Fold_Macro_F1",
    "Mean_Fold_Accuracy"
  ),
  value = c(
    mean(class_f1),
    sum(true_positive) /
      sum(svm_selected_confusion_matrix),
    svm_linear_selected$results$Macro_F1,
    svm_linear_selected$results$Accuracy
  ),
  row.names = NULL
)

cat(
  "\nSelected weighted Linear SVM:",
  "\nC:", selected_weighted_C,
  "\nClass weights:\n"
)

print(svm_class_weight_table)

cat("\nOut-of-fold confusion matrix:\n")
print(svm_selected_confusion_matrix)

cat("\nOut-of-fold class metrics:\n")
print(svm_selected_class_metrics)

cat("\nOverall out-of-fold metrics:\n")
print(svm_selected_pooled_metrics)

# -------------------------------------------------------------------
# Phase 5: model specification comparison
# -------------------------------------------------------------------

# Compare the selected weighted model with both the majority-class
# baseline and the unweighted Linear SVM. All SVM metrics are based on
# the same provider-grouped cross-validation folds.

svm_model_comparison <- data.frame(
  specification = c(
    "Majority-class baseline",
    "Unweighted Linear SVM",
    "Weighted Linear SVM"
  ),
  C = c(
    NA_real_,
    unweighted_reference_C,
    selected_weighted_C
  ),
  weighting = c(
    "None",
    "None",
    "Square-root inverse frequency"
  ),
  Macro_F1 = c(
    unname(baseline_metrics["Macro_F1"]),
    unweighted_reference_result$Macro_F1,
    svm_linear_selected$results$Macro_F1
  ),
  Accuracy = c(
    unname(baseline_metrics["Accuracy"]),
    unweighted_reference_result$Accuracy,
    svm_linear_selected$results$Accuracy
  ),
  Recall_Below_NQS = c(
    unname(baseline_metrics["Recall_Below.NQS"]),
    unweighted_reference_result$Recall_Below.NQS,
    svm_linear_selected$results$Recall_Below.NQS
  ),
  Recall_Meeting_NQS = c(
    unname(baseline_metrics["Recall_Meeting.NQS"]),
    unweighted_reference_result$Recall_Meeting.NQS,
    svm_linear_selected$results$Recall_Meeting.NQS
  ),
  Recall_Above_NQS = c(
    unname(baseline_metrics["Recall_Above.NQS"]),
    unweighted_reference_result$Recall_Above.NQS,
    svm_linear_selected$results$Recall_Above.NQS
  ),
  row.names = NULL
)

svm_model_comparison$Macro_F1_change_vs_baseline <-
  svm_model_comparison$Macro_F1 -
  svm_model_comparison$Macro_F1[1]

svm_model_comparison$Accuracy_change_vs_baseline <-
  svm_model_comparison$Accuracy -
  svm_model_comparison$Accuracy[1]

cat("\nLinear SVM specification comparison:\n")
print(svm_model_comparison)

# -------------------------------------------------------------------
# Phase 5: tuning plot
# -------------------------------------------------------------------

svm_tuning_plot_data <- rbind(
  data.frame(
    specification = "Unweighted",
    svm_linear_unweighted_results
  ),
  data.frame(
    specification = "Weighted",
    svm_linear_weighted_results
  )
)

svm_tuning_plot_data$specification <- factor(
  svm_tuning_plot_data$specification,
  levels = c("Unweighted", "Weighted")
)

svm_tuning_plot <- ggplot(
  svm_tuning_plot_data,
  aes(
    x = log2(C),
    y = Macro_F1,
    colour = specification,
    group = specification
  )
) +
  geom_errorbar(
    aes(
      ymin = Macro_F1 - Macro_F1SD,
      ymax = Macro_F1 + Macro_F1SD
    ),
    width = 0.10,
    alpha = 0.45
  ) +
  geom_line(
    linewidth = 0.7
  ) +
  geom_point(
    size = 2.4
  ) +
  geom_vline(
    xintercept = log2(selected_weighted_C),
    linetype = "dashed",
    colour = "grey35"
  ) +
  scale_x_continuous(
    breaks = sort(
      unique(log2(svm_tuning_plot_data$C))
    )
  ) +
  labs(
    title = "Linear SVM cross-validation performance",
    subtitle = paste0(
      "Dashed line: selected weighted C = ",
      selected_weighted_C
    ),
    x = "log2(C)",
    y = "Mean cross-validation Macro-F1",
    colour = "Specification"
  ) +
  theme_minimal(
    base_size = 11
  ) +
  theme(
    legend.position = "bottom",
    plot.title.position = "plot"
  )

print(svm_tuning_plot)

# -------------------------------------------------------------------
# Phase 6: save reproducible outputs
# -------------------------------------------------------------------

dir.create(
  "outputs/models/svm",
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  "figures/models",
  recursive = TRUE,
  showWarnings = FALSE
)

ggsave(
  filename = "figures/models/svm_linear_tuning.png",
  plot = svm_tuning_plot,
  width = 7,
  height = 5,
  units = "in",
  dpi = 300
)

cat(
  "\nSaved tuning plot:",
  file.exists("figures/models/svm_linear_tuning.png"),
  "\n"
)

# -------------------------------------------------------------------
# Phase 6: save tables and reproducibility information
# -------------------------------------------------------------------

svm_confusion_output <- as.data.frame.matrix(
  svm_selected_confusion_matrix
)

svm_confusion_output <- data.frame(
  Observed = rownames(svm_confusion_output),
  svm_confusion_output,
  row.names = NULL,
  check.names = FALSE
)

svm_best_parameters <- data.frame(
  kernel = "Linear",
  C = selected_weighted_C,
  weighting = "Square-root inverse frequency",
  weight_Below_NQS = unname(
    svm_class_weights["Below NQS"]
  ),
  weight_Meeting_NQS = unname(
    svm_class_weights["Meeting NQS"]
  ),
  weight_Above_NQS = unname(
    svm_class_weights["Above NQS"]
  ),
  primary_metric = "Macro_F1",
  row.names = NULL
)

elapsed_seconds <- function(timing_object) {
  as.numeric(timing_object["elapsed"])
}

svm_timing_results <- data.frame(
  specification = c(
    "Unweighted pilot",
    rep("Unweighted", 5),
    rep("Weighted", 6)
  ),
  C = c(
    1,
    0.0078125,
    0.03125,
    0.125,
    0.5,
    2,
    0.0078125,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5
  ),
  elapsed_seconds = c(
    elapsed_seconds(svm_linear_pilot_time),
    elapsed_seconds(svm_linear_C_00078125_time),
    elapsed_seconds(svm_linear_C_003125_time),
    elapsed_seconds(svm_linear_C_0125_time),
    elapsed_seconds(svm_linear_C_05_time),
    elapsed_seconds(svm_linear_C_2_time),
    elapsed_seconds(svm_linear_weighted_C_00078125_time),
    elapsed_seconds(svm_linear_weighted_C_003125_time),
    elapsed_seconds(svm_linear_weighted_C_00625_time),
    elapsed_seconds(svm_linear_weighted_C_0125_time),
    elapsed_seconds(svm_linear_weighted_C_025_time),
    elapsed_seconds(svm_linear_weighted_C_05_time)
  ),
  row.names = NULL
)

svm_package_versions <- data.frame(
  package = c(
    "R",
    "caret",
    "recipes",
    "kernlab",
    "ggplot2"
  ),
  version = c(
    as.character(getRversion()),
    as.character(packageVersion("caret")),
    as.character(packageVersion("recipes")),
    as.character(packageVersion("kernlab")),
    as.character(packageVersion("ggplot2"))
  ),
  row.names = NULL
)

write.csv(
  svm_tuning_plot_data,
  "outputs/models/svm/svm_linear_tuning_results.csv",
  row.names = FALSE
)

write.csv(
  weighted_resample_results,
  "outputs/models/svm/svm_linear_resample_results.csv",
  row.names = FALSE
)

write.csv(
  svm_best_parameters,
  "outputs/models/svm/svm_linear_best_parameters.csv",
  row.names = FALSE
)

write.csv(
  svm_selected_class_metrics,
  "outputs/models/svm/svm_linear_class_metrics.csv",
  row.names = FALSE
)

write.csv(
  svm_confusion_output,
  "outputs/models/svm/svm_linear_confusion_matrix.csv",
  row.names = FALSE
)

write.csv(
  svm_selected_pooled_metrics,
  "outputs/models/svm/svm_linear_overall_metrics.csv",
  row.names = FALSE
)

write.csv(
  svm_model_comparison,
  "outputs/models/svm/svm_linear_model_comparison.csv",
  row.names = FALSE
)

write.csv(
  svm_timing_results,
  "outputs/models/svm/svm_linear_timing.csv",
  row.names = FALSE
)

write.csv(
  svm_package_versions,
  "outputs/models/svm/svm_linear_package_versions.csv",
  row.names = FALSE
)

writeLines(
  capture.output(sessionInfo()),
  "outputs/models/svm/svm_linear_session_info.txt"
)

svm_output_files <- c(
  "outputs/models/svm/svm_linear_tuning_results.csv",
  "outputs/models/svm/svm_linear_resample_results.csv",
  "outputs/models/svm/svm_linear_best_parameters.csv",
  "outputs/models/svm/svm_linear_class_metrics.csv",
  "outputs/models/svm/svm_linear_confusion_matrix.csv",
  "outputs/models/svm/svm_linear_overall_metrics.csv",
  "outputs/models/svm/svm_linear_model_comparison.csv",
  "outputs/models/svm/svm_linear_timing.csv",
  "outputs/models/svm/svm_linear_package_versions.csv",
  "outputs/models/svm/svm_linear_session_info.txt",
  "figures/models/svm_linear_tuning.png"
)

svm_output_check <- data.frame(
  file = svm_output_files,
  exists = file.exists(svm_output_files),
  row.names = NULL
)

stopifnot(
  all(svm_output_check$exists)
)

cat("\nSaved SVM outputs:\n")
print(svm_output_check)


# -------------------------------------------------------------------
# Phase 7: optional RBF SVM pilot
# -------------------------------------------------------------------

# Test one weighted RBF specification to assess whether a nonlinear
# decision boundary provides enough benefit to justify further tuning.
# The preprocessing, CV folds, metric and class weights remain unchanged.

rbf_pilot_sigma <- 1 / (ncol(scaled_validation_check) - 1L)
rbf_pilot_C <- selected_weighted_C

stopifnot(
  rbf_pilot_sigma > 0,
  rbf_pilot_C > 0
)

set.seed(5018L)

svm_rbf_pilot_time <- system.time({
  svm_rbf_pilot <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmRadial",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(
      sigma = rbf_pilot_sigma,
      C = rbf_pilot_C
    ),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nWeighted RBF SVM pilot completed.",
  "\nC:", rbf_pilot_C,
  "\nsigma:", rbf_pilot_sigma,
  "\n\nCross-validation results:\n"
)

print(svm_rbf_pilot$results)

cat("\nElapsed time:\n")
print(svm_rbf_pilot_time)


# RBF candidate with a more localised decision boundary:
# sigma = 0.0625, while holding C and all other settings constant.

set.seed(5019L)

svm_rbf_sigma_00625_time <- system.time({
  svm_rbf_sigma_00625 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmRadial",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(
      sigma = 0.0625,
      C = rbf_pilot_C
    ),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nWeighted RBF SVM candidate completed.",
  "\nC:", rbf_pilot_C,
  "\nsigma:", 0.0625,
  "\n\nCross-validation results:\n"
)

print(svm_rbf_sigma_00625$results)

cat("\nElapsed time:\n")
print(svm_rbf_sigma_00625_time)

# Final RBF sigma sensitivity candidate:
# sigma = 0.015625, with C and all other settings unchanged.

set.seed(5020L)

svm_rbf_sigma_0015625_time <- system.time({
  svm_rbf_sigma_0015625 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmRadial",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(
      sigma = 0.015625,
      C = rbf_pilot_C
    ),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nWeighted RBF SVM candidate completed.",
  "\nC:", rbf_pilot_C,
  "\nsigma:", 0.015625,
  "\n\nCross-validation results:\n"
)

print(svm_rbf_sigma_0015625$results)

cat("\nElapsed time:\n")
print(svm_rbf_sigma_0015625_time)

# -------------------------------------------------------------------
# Phase 8: RBF pilot summary and diagnostics
# -------------------------------------------------------------------

selected_rbf_C <- 0.125
selected_rbf_sigma <- 0.015625
selected_rbf_model <- svm_rbf_sigma_0015625

# Combine the three RBF sensitivity-analysis candidates.

rbf_result_columns <- c(
  "sigma",
  "C",
  "Macro_F1",
  "Macro_F1SD",
  "Accuracy",
  "AccuracySD",
  "Recall_Below.NQS",
  "Recall_Meeting.NQS",
  "Recall_Above.NQS"
)

svm_rbf_pilot_results <- rbind(
  svm_rbf_sigma_0015625$results[, rbf_result_columns],
  svm_rbf_pilot$results[, rbf_result_columns],
  svm_rbf_sigma_00625$results[, rbf_result_columns]
)

svm_rbf_pilot_results <-
  svm_rbf_pilot_results[
    order(svm_rbf_pilot_results$sigma),
  ]

rownames(svm_rbf_pilot_results) <- NULL

rbf_selected_result <-
  svm_rbf_pilot_results[
    svm_rbf_pilot_results$C == selected_rbf_C &
      svm_rbf_pilot_results$sigma == selected_rbf_sigma,
  ]

stopifnot(
  nrow(rbf_selected_result) == 1L,
  rbf_selected_result$Macro_F1 ==
    max(svm_rbf_pilot_results$Macro_F1)
)

cat("\nRBF pilot tuning results:\n")
print(svm_rbf_pilot_results)

cat(
  "\nSelected RBF pilot configuration:",
  "\nC:", selected_rbf_C,
  "\nsigma:", selected_rbf_sigma,
  "\n"
)


# -------------------------------------------------------------------
# Phase 8: provisional RBF sigma selection
# -------------------------------------------------------------------

# The initial RBF sensitivity analysis compared three sigma values
# while holding C constant at 0.125. Among these candidates,
# sigma = 0.015625 achieved the highest mean CV Macro-F1.
#
# This was only a provisional selection because C had not yet been
# evaluated for the RBF kernel. The provisional diagnostics originally
# placed in this phase were subsequently discarded after Phase 9
# examined additional C values and selected C = 0.5.
#
# Final out-of-fold diagnostics and the Linear-RBF comparison are
# therefore calculated only once in Phase 10 using the final targeted
# configuration:
#   sigma = 0.015625
#   C = 0.5

selected_rbf_sigma <- 0.015625

# -------------------------------------------------------------------
# Phase 9: targeted RBF C sensitivity analysis
# -------------------------------------------------------------------

# Lower-C candidate at the selected pilot sigma.
# Only C changes; preprocessing, folds, weights and sigma remain fixed.

set.seed(5021L)

svm_rbf_C_003125_time <- system.time({
  svm_rbf_C_003125 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmRadial",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(
      sigma = selected_rbf_sigma,
      C = 0.03125
    ),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nWeighted RBF SVM C-sensitivity candidate completed.",
  "\nC:", 0.03125,
  "\nsigma:", selected_rbf_sigma,
  "\n\nCross-validation results:\n"
)

print(svm_rbf_C_003125$results)

cat("\nElapsed time:\n")
print(svm_rbf_C_003125_time)

# Higher-C candidate at the selected pilot sigma.
# Only C changes; preprocessing, folds, weights and sigma remain fixed.

set.seed(5022L)

svm_rbf_C_05_time <- system.time({
  svm_rbf_C_05 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmRadial",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(
      sigma = selected_rbf_sigma,
      C = 0.5
    ),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nWeighted RBF SVM C-sensitivity candidate completed.",
  "\nC:", 0.5,
  "\nsigma:", selected_rbf_sigma,
  "\n\nCross-validation results:\n"
)

print(svm_rbf_C_05$results)

cat("\nElapsed time:\n")
print(svm_rbf_C_05_time)

# Extended upper-bound C candidate at the selected RBF sigma.
# Only C changes; preprocessing, folds, weights and sigma remain fixed.

set.seed(5023L)

svm_rbf_C_2_time <- system.time({
  svm_rbf_C_2 <- train(
    scaled_recipe,
    data = training_model_data,
    method = "svmRadial",
    trControl = coarse_control_verbose,
    tuneGrid = data.frame(
      sigma = selected_rbf_sigma,
      C = 2
    ),
    metric = "Macro_F1",
    maximize = TRUE,
    class.weights = svm_class_weights
  )
})

cat(
  "\nWeighted RBF SVM extended C candidate completed.",
  "\nC:", 2,
  "\nsigma:", selected_rbf_sigma,
  "\n\nCross-validation results:\n"
)

print(svm_rbf_C_2$results)

cat("\nElapsed time:\n")
print(svm_rbf_C_2_time)


# -------------------------------------------------------------------
# Phase 10: final targeted RBF selection and diagnostics
# -------------------------------------------------------------------

# Final RBF configuration selected by mean cross-validated Macro-F1.

selected_rbf_C <- 0.5
selected_rbf_sigma <- 0.015625
selected_rbf_model <- svm_rbf_C_05

# Combine all six unique RBF candidates.

rbf_result_columns <- c(
  "sigma",
  "C",
  "Macro_F1",
  "Macro_F1SD",
  "Accuracy",
  "AccuracySD",
  "Recall_Below.NQS",
  "Recall_Meeting.NQS",
  "Recall_Above.NQS"
)

svm_rbf_targeted_results <- rbind(
  svm_rbf_C_003125$results[, rbf_result_columns],
  svm_rbf_sigma_0015625$results[, rbf_result_columns],
  svm_rbf_C_05$results[, rbf_result_columns],
  svm_rbf_C_2$results[, rbf_result_columns],
  svm_rbf_pilot$results[, rbf_result_columns],
  svm_rbf_sigma_00625$results[, rbf_result_columns]
)

svm_rbf_targeted_results <-
  svm_rbf_targeted_results[
    order(
      svm_rbf_targeted_results$sigma,
      svm_rbf_targeted_results$C
    ),
  ]

rownames(svm_rbf_targeted_results) <- NULL

svm_rbf_targeted_results$search_role <- ifelse(
  svm_rbf_targeted_results$sigma == 0.015625 &
    svm_rbf_targeted_results$C == 0.125,
  "Shared reference",
  ifelse(
    svm_rbf_targeted_results$sigma == 0.015625,
    "C sensitivity",
    "Sigma sensitivity"
  )
)

rbf_selected_result <-
  svm_rbf_targeted_results[
    svm_rbf_targeted_results$C == selected_rbf_C &
      svm_rbf_targeted_results$sigma == selected_rbf_sigma,
  ]

rbf_parameter_pairs <- paste(
  svm_rbf_targeted_results$sigma,
  svm_rbf_targeted_results$C,
  sep = "_"
)

stopifnot(
  nrow(svm_rbf_targeted_results) == 6L,
  !anyDuplicated(rbf_parameter_pairs),
  nrow(rbf_selected_result) == 1L,
  rbf_selected_result$Macro_F1 ==
    max(svm_rbf_targeted_results$Macro_F1)
)

cat("\nFinal targeted RBF results:\n")
print(svm_rbf_targeted_results)

cat(
  "\nSelected weighted RBF SVM:",
  "\nC:", selected_rbf_C,
  "\nsigma:", selected_rbf_sigma,
  "\n"
)

# -------------------------------------------------------------------
# Final selected RBF out-of-fold diagnostics
# -------------------------------------------------------------------

svm_rbf_selected_predictions <- selected_rbf_model$pred
outcome_levels <- levels(training_model_data$rating_tier)

predictions_per_row <- table(
  svm_rbf_selected_predictions$rowIndex
)

stopifnot(
  nrow(svm_rbf_selected_predictions) ==
    nrow(training_model_data),
  setequal(
    svm_rbf_selected_predictions$rowIndex,
    seq_len(nrow(training_model_data))
  ),
  all(predictions_per_row == 1L),
  !anyNA(
    svm_rbf_selected_predictions[
      ,
      c("pred", "obs", "rowIndex", "Resample", "sigma", "C")
    ]
  ),
  all(svm_rbf_selected_predictions$C == selected_rbf_C),
  all(
    svm_rbf_selected_predictions$sigma ==
      selected_rbf_sigma
  )
)

svm_rbf_confusion_matrix <- table(
  Observed = factor(
    svm_rbf_selected_predictions$obs,
    levels = outcome_levels
  ),
  Predicted = factor(
    svm_rbf_selected_predictions$pred,
    levels = outcome_levels
  )
)

rbf_true_positive <- diag(svm_rbf_confusion_matrix)

rbf_precision <- ifelse(
  colSums(svm_rbf_confusion_matrix) == 0,
  0,
  rbf_true_positive /
    colSums(svm_rbf_confusion_matrix)
)

rbf_recall <- ifelse(
  rowSums(svm_rbf_confusion_matrix) == 0,
  0,
  rbf_true_positive /
    rowSums(svm_rbf_confusion_matrix)
)

rbf_class_F1 <- ifelse(
  rbf_precision + rbf_recall == 0,
  0,
  2 * rbf_precision * rbf_recall /
    (rbf_precision + rbf_recall)
)

svm_rbf_class_metrics <- data.frame(
  rating_tier = outcome_levels,
  support = as.integer(
    rowSums(svm_rbf_confusion_matrix)
  ),
  precision = as.numeric(rbf_precision),
  recall = as.numeric(rbf_recall),
  F1 = as.numeric(rbf_class_F1),
  row.names = NULL
)

svm_rbf_pooled_metrics <- data.frame(
  metric = c(
    "Pooled_OOF_Macro_F1",
    "Pooled_OOF_Accuracy",
    "Mean_Fold_Macro_F1",
    "Mean_Fold_Accuracy"
  ),
  value = c(
    mean(rbf_class_F1),
    sum(rbf_true_positive) /
      sum(svm_rbf_confusion_matrix),
    rbf_selected_result$Macro_F1,
    rbf_selected_result$Accuracy
  )
)

cat("\nFinal RBF out-of-fold confusion matrix:\n")
print(svm_rbf_confusion_matrix)

cat("\nFinal RBF out-of-fold class metrics:\n")
print(svm_rbf_class_metrics)

cat("\nFinal RBF overall metrics:\n")
print(svm_rbf_pooled_metrics)

# -------------------------------------------------------------------
# Final Linear and RBF specification comparison
# -------------------------------------------------------------------

svm_kernel_comparison <- svm_model_comparison
svm_kernel_comparison$sigma <- NA_real_

svm_kernel_comparison <- svm_kernel_comparison[
  ,
  c(
    "specification",
    "C",
    "sigma",
    setdiff(
      names(svm_kernel_comparison),
      c("specification", "C", "sigma")
    )
  )
]

rbf_comparison_row <- svm_kernel_comparison[1, ]
rbf_comparison_row[1, ] <- NA

rbf_comparison_row$specification <-
  "Weighted RBF SVM (targeted)"
rbf_comparison_row$C <- selected_rbf_C
rbf_comparison_row$sigma <- selected_rbf_sigma
rbf_comparison_row$weighting <-
  "Square-root inverse frequency"
rbf_comparison_row$Macro_F1 <-
  rbf_selected_result$Macro_F1
rbf_comparison_row$Accuracy <-
  rbf_selected_result$Accuracy
rbf_comparison_row$Recall_Below_NQS <-
  rbf_selected_result$Recall_Below.NQS
rbf_comparison_row$Recall_Meeting_NQS <-
  rbf_selected_result$Recall_Meeting.NQS
rbf_comparison_row$Recall_Above_NQS <-
  rbf_selected_result$Recall_Above.NQS

baseline_comparison_row <-
  svm_kernel_comparison[
    svm_kernel_comparison$specification ==
      "Majority-class baseline",
  ]

rbf_comparison_row$Macro_F1_change_vs_baseline <-
  rbf_comparison_row$Macro_F1 -
  baseline_comparison_row$Macro_F1

rbf_comparison_row$Accuracy_change_vs_baseline <-
  rbf_comparison_row$Accuracy -
  baseline_comparison_row$Accuracy

svm_kernel_comparison <- rbind(
  svm_kernel_comparison,
  rbf_comparison_row
)

rownames(svm_kernel_comparison) <- NULL

cat("\nFinal Linear and RBF SVM comparison:\n")
print(svm_kernel_comparison)


# -------------------------------------------------------------------
# Phase 11: save final RBF outputs
# -------------------------------------------------------------------

dir.create(
  "outputs/models/svm",
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  "figures/models",
  recursive = TRUE,
  showWarnings = FALSE
)

# Selected parameters

svm_rbf_best_parameters <- data.frame(
  method = "svmRadial",
  kernel = "RBF",
  C = selected_rbf_C,
  sigma = selected_rbf_sigma,
  weighting = "Square-root inverse frequency",
  selection_metric = "Macro_F1",
  tuning_strategy =
    "Targeted two-parameter sensitivity analysis",
  cross_validation =
    "5-fold provider-grouped cross-validation"
)

# Selected fold-level results

svm_rbf_selected_resamples <- selected_rbf_model$resample
svm_rbf_selected_resamples$C <- selected_rbf_C
svm_rbf_selected_resamples$sigma <- selected_rbf_sigma

# Confusion matrix in long-form CSV format

svm_rbf_confusion_output <- as.data.frame(
  svm_rbf_confusion_matrix
)

names(svm_rbf_confusion_output) <- c(
  "Observed",
  "Predicted",
  "n"
)

# Training-time summary

svm_rbf_timing <- data.frame(
  sigma = c(
    0.015625,
    0.015625,
    0.015625,
    0.015625,
    0.03125,
    0.0625
  ),
  C = c(
    0.03125,
    0.125,
    0.5,
    2,
    0.125,
    0.125
  ),
  user_seconds = c(
    unname(svm_rbf_C_003125_time["user.self"]),
    unname(svm_rbf_sigma_0015625_time["user.self"]),
    unname(svm_rbf_C_05_time["user.self"]),
    unname(svm_rbf_C_2_time["user.self"]),
    unname(svm_rbf_pilot_time["user.self"]),
    unname(svm_rbf_sigma_00625_time["user.self"])
  ),
  system_seconds = c(
    unname(svm_rbf_C_003125_time["sys.self"]),
    unname(svm_rbf_sigma_0015625_time["sys.self"]),
    unname(svm_rbf_C_05_time["sys.self"]),
    unname(svm_rbf_C_2_time["sys.self"]),
    unname(svm_rbf_pilot_time["sys.self"]),
    unname(svm_rbf_sigma_00625_time["sys.self"])
  ),
  elapsed_seconds = c(
    unname(svm_rbf_C_003125_time["elapsed"]),
    unname(svm_rbf_sigma_0015625_time["elapsed"]),
    unname(svm_rbf_C_05_time["elapsed"]),
    unname(svm_rbf_C_2_time["elapsed"]),
    unname(svm_rbf_pilot_time["elapsed"]),
    unname(svm_rbf_sigma_00625_time["elapsed"])
  )
)

# Save CSV outputs

write.csv(
  svm_rbf_targeted_results,
  "outputs/models/svm/svm_rbf_targeted_tuning_results.csv",
  row.names = FALSE
)

write.csv(
  svm_rbf_selected_resamples,
  "outputs/models/svm/svm_rbf_selected_resample_results.csv",
  row.names = FALSE
)

write.csv(
  svm_rbf_best_parameters,
  "outputs/models/svm/svm_rbf_best_parameters.csv",
  row.names = FALSE
)

write.csv(
  svm_rbf_class_metrics,
  "outputs/models/svm/svm_rbf_class_metrics.csv",
  row.names = FALSE
)

write.csv(
  svm_rbf_confusion_output,
  "outputs/models/svm/svm_rbf_confusion_matrix.csv",
  row.names = FALSE
)

write.csv(
  svm_rbf_pooled_metrics,
  "outputs/models/svm/svm_rbf_overall_metrics.csv",
  row.names = FALSE
)

write.csv(
  svm_kernel_comparison,
  "outputs/models/svm/svm_kernel_model_comparison.csv",
  row.names = FALSE
)

write.csv(
  svm_rbf_timing,
  "outputs/models/svm/svm_rbf_timing.csv",
  row.names = FALSE
)

# Prepare two-panel tuning plot data

rbf_C_plot_data <- svm_rbf_targeted_results[
  svm_rbf_targeted_results$sigma ==
    selected_rbf_sigma,
]

rbf_C_plot_data$panel <-
  paste0("C sensitivity (sigma = ", selected_rbf_sigma, ")")

rbf_C_plot_data$parameter_value <-
  rbf_C_plot_data$C

rbf_C_plot_data$selected <-
  rbf_C_plot_data$C == selected_rbf_C

rbf_sigma_plot_data <- svm_rbf_targeted_results[
  svm_rbf_targeted_results$C == 0.125,
]

rbf_sigma_plot_data$panel <-
  "Sigma sensitivity (C = 0.125)"

rbf_sigma_plot_data$parameter_value <-
  rbf_sigma_plot_data$sigma

rbf_sigma_plot_data$selected <- FALSE

rbf_tuning_plot_data <- rbind(
  rbf_C_plot_data,
  rbf_sigma_plot_data
)

svm_rbf_tuning_plot <- ggplot(
  rbf_tuning_plot_data,
  aes(
    x = log2(parameter_value),
    y = Macro_F1,
    group = 1
  )
) +
  geom_errorbar(
    aes(
      ymin = Macro_F1 - Macro_F1SD,
      ymax = Macro_F1 + Macro_F1SD
    ),
    width = 0.12,
    colour = "grey55"
  ) +
  geom_line(
    colour = "#35608D",
    linewidth = 0.7
  ) +
  geom_point(
    aes(colour = selected),
    size = 2.8
  ) +
  facet_wrap(
    ~panel,
    scales = "free_x"
  ) +
  scale_colour_manual(
    values = c(
      "FALSE" = "#35608D",
      "TRUE" = "#C44E52"
    ),
    guide = "none"
  ) +
  scale_x_continuous(
    breaks = sort(
      unique(log2(rbf_tuning_plot_data$parameter_value))
    ),
    labels = function(x) format(2^x, trim = TRUE)
  ) +
  labs(
    x = "Parameter value",
    y = "Mean cross-validated Macro-F1",
    title = "Weighted RBF SVM targeted sensitivity analysis",
    subtitle = "Error bars show ±1 fold-level standard deviation"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold")
  )

ggsave(
  filename = "figures/models/svm_rbf_targeted_tuning.png",
  plot = svm_rbf_tuning_plot,
  width = 9,
  height = 4.8,
  dpi = 300,
  bg = "white"
)

# Verify saved outputs

svm_rbf_output_files <- c(
  "outputs/models/svm/svm_rbf_targeted_tuning_results.csv",
  "outputs/models/svm/svm_rbf_selected_resample_results.csv",
  "outputs/models/svm/svm_rbf_best_parameters.csv",
  "outputs/models/svm/svm_rbf_class_metrics.csv",
  "outputs/models/svm/svm_rbf_confusion_matrix.csv",
  "outputs/models/svm/svm_rbf_overall_metrics.csv",
  "outputs/models/svm/svm_kernel_model_comparison.csv",
  "outputs/models/svm/svm_rbf_timing.csv",
  "figures/models/svm_rbf_targeted_tuning.png"
)

svm_rbf_output_check <- data.frame(
  file = svm_rbf_output_files,
  exists = file.exists(svm_rbf_output_files)
)

cat("\nSaved RBF outputs:\n")
print(svm_rbf_output_check)

stopifnot(all(svm_rbf_output_check$exists))
