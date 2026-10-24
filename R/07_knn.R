# Run from the repository root. Uses the group's exact provider split/folds.
# Test outcomes are deliberately not evaluated during model development.
required_files <- c("R/06_model_setup.R", "data/processed/acecqa_analysis.csv",
                    "data/processed/acecqa_variable_dictionary.csv")
if (!all(file.exists(required_files))) {
  stop("Run from the group repository root with shared setup and processed CSV files present.")
}
if (dir.exists(".R-library")) .libPaths(c(normalizePath(".R-library"), .libPaths()))
source("R/06_model_setup.R")
if (!requireNamespace("kknn", quietly = TRUE)) stop("Install package kknn first.")
dir.create("outputs/knn", recursive = TRUE, showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)
out <- "outputs/knn"
set.seed(5005)

# Audit the shared split and model inputs without using test outcomes for tuning.
stopifnot(length(intersect(training_data$provider_id, test_data$provider_id)) == 0L,
          identical(training_model_data$rating_tier, training_data$rating_tier),
          !any(grepl("(^|_)(rating|quality|qa[1-7])(_|$)|^service_id$|^provider_id$",
                     predictor_names, ignore.case = TRUE)),
          length(cv_index) == 5L, identical(names(cv_index), names(cv_index_out)))
audit_rows <- list()
for (f in names(cv_index)) {
  a <- cv_index[[f]]; v <- cv_index_out[[f]]
  stopifnot(length(intersect(a, v)) == 0L,
            setequal(c(a, v), seq_len(nrow(training_model_data))),
            length(intersect(training_data$provider_id[a], training_data$provider_id[v])) == 0L,
            all(table(training_model_data$rating_tier[a]) > 0),
            all(table(training_model_data$rating_tier[v]) > 0))
  audit_rows[[f]] <- data.frame(fold = f, training_rows = length(a), validation_rows = length(v),
    provider_overlap = 0L, all_classes_present = TRUE)
}
readr::write_csv(dplyr::bind_rows(audit_rows), file.path(out, "leakage_audit.csv"))

# Preprocess separately inside each shared training fold. kknn's internal
# scaling is disabled because the shared recipe already scales numerics.
fold_data <- lapply(names(cv_index), function(f) {
  rec <- recipes::prep(scaled_recipe, training = training_model_data[cv_index[[f]], ])
  list(train = recipes::bake(rec, new_data = training_model_data[cv_index[[f]], ]),
       valid = recipes::bake(rec, new_data = training_model_data[cv_index_out[[f]], ]))
})
names(fold_data) <- names(cv_index)
stopifnot(all(vapply(fold_data, function(x) !anyNA(x$train) && !anyNA(x$valid), logical(1))))
stopifnot(all(vapply(fold_data, function(x) {
  identical(names(x$train), names(x$valid)) &&
    all(vapply(x$train[setdiff(names(x$train), "rating_tier")], is.numeric, logical(1)))
}, logical(1))))

grid <- expand.grid(k = c(3L, 5L, 9L, 15L, 25L, 41L, 61L, 81L, 121L, 161L),
                    kernel = c("rectangular", "triangular"), stringsAsFactors = FALSE)
results <- list()
predictions <- list()
evaluate_grid <- function(candidates) {
  for (i in seq_len(nrow(candidates))) {
    k <- candidates$k[i]; kernel <- candidates$kernel[i]
    for (f in names(fold_data)) {
      message("k=", k, ", kernel=", kernel, ", ", f)
      x <- fold_data[[f]]
      fit <- kknn::kknn(rating_tier ~ ., train = as.data.frame(x$train),
                        test = as.data.frame(x$valid), k = k, distance = 2,
                        kernel = kernel, scale = FALSE)
      pred <- factor(fit$fitted.values, levels = levels(training_model_data$rating_tier))
      obs <- x$valid$rating_tier
      metrics <- macro_f1_summary(data.frame(obs = obs, pred = pred))
      results[[length(results) + 1L]] <<- data.frame(k = k, kernel = kernel, fold = f,
                                                   as.list(metrics), check.names = FALSE)
      predictions[[length(predictions) + 1L]] <<- data.frame(k = k, kernel = kernel,
        fold = f, rowIndex = cv_index_out[[f]], obs = obs, pred = pred)
    }
  }
}
summarise_grid <- function() {
  dplyr::bind_rows(results) |>
    dplyr::group_by(k, kernel) |>
    dplyr::summarise(Macro_F1SD = sd(Macro_F1), Macro_F1 = mean(Macro_F1),
                     AccuracySD = sd(Accuracy), Accuracy = mean(Accuracy), .groups = "drop") |>
    dplyr::arrange(dplyr::desc(Macro_F1), dplyr::desc(k), kernel)
}
evaluate_grid(grid)
summary <- summarise_grid()
# Expand an upper-bound optimum while keeping k valid for every fold.
max_k <- min(vapply(fold_data, function(x) nrow(x$train) - 2L, integer(1)))
while (summary$k[1] == max(grid$k) && max(grid$k) < max_k) {
  next_k <- min(2L * max(grid$k) + 1L, max_k)
  extra <- expand.grid(k = next_k, kernel = c("rectangular", "triangular"), stringsAsFactors = FALSE)
  evaluate_grid(extra)
  grid <- rbind(grid, extra)
  summary <- summarise_grid()
}
best <- summary[1, ]
fold_results <- dplyr::bind_rows(results)
stopifnot(!anyNA(summary), all(table(fold_results$k, fold_results$kernel) == n_folds))
# Compare the best configuration within each kernel using mean fold metrics.
# Recall is averaged across folds, not calculated from pooled predictions.
kernel_winners <- summary |>
  dplyr::group_by(kernel) |>
  dplyr::slice_head(n = 1L) |>
  dplyr::ungroup()
mean_recalls <- fold_results |>
  dplyr::group_by(k, kernel) |>
  dplyr::summarise(dplyr::across(dplyr::starts_with("Recall_"), mean), .groups = "drop")
kernel_comparison <- dplyr::left_join(kernel_winners, mean_recalls, by = c("k", "kernel"))
readr::write_csv(kernel_comparison, file.path(out, "kernel_comparison.csv"))
all_predictions <- dplyr::bind_rows(predictions)
oof <- all_predictions[all_predictions$k == best$k & all_predictions$kernel == best$kernel, ]
oof <- oof[order(oof$rowIndex), ]
stopifnot(identical(oof$rowIndex, seq_len(nrow(training_model_data))))
oof$service_id <- training_data$service_id[oof$rowIndex]
oof$provider_id <- training_data$provider_id[oof$rowIndex]

class_metrics <- function(obs, pred) {
  do.call(rbind, lapply(levels(obs), function(cl) {
    tp <- sum(obs == cl & pred == cl); fp <- sum(obs != cl & pred == cl)
    fn <- sum(obs == cl & pred != cl)
    data.frame(class = cl, support = sum(obs == cl),
      Precision = if (tp + fp == 0) 0 else tp / (tp + fp),
      Recall = if (tp + fn == 0) 0 else tp / (tp + fn),
      F1 = if (2 * tp + fp + fn == 0) 0 else 2 * tp / (2 * tp + fp + fn))
  }))
}
classes <- class_metrics(oof$obs, oof$pred)
cm <- table(Actual = oof$obs, Predicted = oof$pred)
baseline <- dplyr::bind_rows(lapply(names(cv_index), function(f) {
  majority <- names(which.max(table(training_model_data$rating_tier[cv_index[[f]]])))
  obs <- training_model_data$rating_tier[cv_index_out[[f]]]
  pred <- factor(rep(majority, length(obs)), levels = levels(obs))
  data.frame(fold = f, as.list(macro_f1_summary(data.frame(obs = obs, pred = pred))))
}))
readr::write_csv(summary, file.path(out, "cv_summary.csv"))
readr::write_csv(fold_results, file.path(out, "cv_fold_metrics.csv"))
readr::write_csv(oof, file.path(out, "selected_oof_predictions.csv"))
readr::write_csv(classes, file.path(out, "class_metrics.csv"))
readr::write_csv(as.data.frame(cm), file.path(out, "confusion_matrix.csv"))
readr::write_csv(baseline, file.path(out, "baseline_cv.csv"))
readr::write_csv(best, file.path(out, "best_parameters.csv"))
readr::write_csv(split_distribution, file.path(out, "split_distribution.csv"))
readr::write_csv(cv_fold_summary, file.path(out, "fold_distribution.csv"))
readr::write_csv(data.frame(service_id = training_data$service_id,
  provider_id = training_data$provider_id, fold = oof$fold), file.path(out, "training_folds.csv"))

p <- ggplot2::ggplot(summary, ggplot2::aes(k, Macro_F1, colour = kernel)) +
  ggplot2::geom_line() + ggplot2::geom_point() +
  ggplot2::geom_errorbar(ggplot2::aes(ymin = Macro_F1 - Macro_F1SD,
                                    ymax = Macro_F1 + Macro_F1SD), width = 1) +
  ggplot2::labs(x = "Neighbours (k)", y = "Mean CV Macro-F1", colour = "Voting kernel",
    title = "KNN tuning: shared provider-grouped five-fold CV",
    subtitle = "Error bars show one fold standard deviation, not confidence intervals") +
  ggplot2::theme_minimal(base_size = 11)
ggplot2::ggsave("figures/knn_tuning.png", p, width = 8, height = 4.5, dpi = 180)
cm_plot <- as.data.frame(cm)
cm_plot$proportion <- as.vector(prop.table(cm, 1))
p2 <- ggplot2::ggplot(cm_plot, ggplot2::aes(Predicted, Actual, fill = proportion)) +
  ggplot2::geom_tile() +
  ggplot2::geom_text(ggplot2::aes(label = sprintf("%d\n%.1f%%", Freq, 100 * proportion))) +
  ggplot2::scale_fill_gradient(low = "white", high = "#58A6C8", limits = c(0, 1)) +
  ggplot2::labs(title = "Selected KNN: pooled out-of-fold predictions", fill = "Row share") +
  ggplot2::theme_minimal(base_size = 11)
ggplot2::ggsave("figures/knn_confusion.png", p2, width = 7, height = 4.5, dpi = 180)

# Save training data and a full-training recipe for later locked evaluation.
# The future prediction uses this exact training reference and preprocessing.
final_recipe <- recipes::prep(scaled_recipe, training = training_model_data)
final_train <- recipes::bake(final_recipe, new_data = training_model_data)
saveRDS(list(recipe = final_recipe, train = final_train, k = best$k,
             kernel = best$kernel, distance = 2, scale = FALSE,
             levels = levels(training_model_data$rating_tier)), file.path(out, "knn_model.rds"))
writeLines(capture.output(sessionInfo()), file.path(out, "session_info.txt"))
message("Completed. Test set was not evaluated. Selected configuration:")
print(best)
