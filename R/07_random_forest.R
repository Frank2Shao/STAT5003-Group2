# Run from the repository root: source("R/07_random_forest.R")
if (!requireNamespace("ranger", quietly = TRUE)) stop("Install ranger first.")
source("R/06_model_setup.R")
dir.create("outputs/random_forest", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/random_forest", recursive = TRUE, showWarnings = FALSE)
out <- "outputs/random_forest"
lev <- levels(training_model_data$rating_tier)
grid <- expand.grid(mtry = c(5L, 12L), min_node_size = c(5L, 20L),
                    weighting = c("uniform", "balanced"), stringsAsFactors = FALSE)
grid$candidate <- seq_len(nrow(grid))
grid$num_trees <- 500L
write_csv(grid, file.path(out, "tuning_grid.csv"))
weights_for <- function(y, mode) {
  if (mode == "uniform") return(rep(1, length(lev)))
  counts <- table(factor(y, levels = lev))
  stopifnot(all(counts > 0))
  as.numeric(length(y) / (length(lev) * counts))
}
fit_rf <- function(x, config, seed, importance = "none") {
  ranger::ranger(rating_tier ~ ., data = as.data.frame(x),
    num.trees = config$num_trees, mtry = config$mtry,
    min.node.size = config$min_node_size, splitrule = "gini",
    class.weights = weights_for(x$rating_tier, config$weighting),
    importance = importance, seed = seed, num.threads = 2L)
}
scores <- list()
predictions <- list()
baseline <- list()
removed <- list()
for (fold in names(cv_index)) {
  tr <- cv_index[[fold]]
  va <- cv_index_out[[fold]]
  fitted_recipe <- prep(tree_recipe, training = training_model_data[tr, ])
  xtr <- juice(fitted_recipe)
  xva <- bake(fitted_recipe, new_data = training_model_data[va, ])
  stopifnot(!anyNA(xtr), !anyNA(xva), max(grid$mtry) < ncol(xtr))
  removed[[fold]] <- tidy(fitted_recipe, number = length(fitted_recipe$steps)) |>
    mutate(fold = fold)
  majority <- names(which.max(table(xtr$rating_tier)))
  base_pred <- factor(rep(majority, length(va)), levels = lev)
  baseline[[fold]] <- data.frame(fold = fold,
    as.list(macro_f1_summary(data.frame(obs = xva$rating_tier, pred = base_pred), lev)))
  for (i in seq_len(nrow(grid))) {
    cat("Fitting", fold, "candidate", i, "of", nrow(grid), "\n")
    fit <- fit_rf(xtr, grid[i, ], 7000L + match(fold, names(cv_index)))
    pred <- factor(predict(fit, data = as.data.frame(xva))$predictions, levels = lev)
    stopifnot(!anyNA(pred))
    key <- paste(fold, i, sep = "_")
    scores[[key]] <- data.frame(candidate = i, fold = fold,
      as.list(macro_f1_summary(data.frame(obs = xva$rating_tier, pred = pred), lev)))
    predictions[[key]] <- data.frame(candidate = i, fold = fold, row = va,
      service_id = training_data$service_id[va],
      provider_id = training_data$provider_id[va], obs = xva$rating_tier, pred = pred)
  }
}
fold_scores <- bind_rows(scores)
summary_scores <- fold_scores |>
  group_by(candidate) |>
  summarise(mean_macro_f1 = mean(Macro_F1), sd_macro_f1 = sd(Macro_F1),
            mean_accuracy = mean(Accuracy), .groups = "drop") |>
  left_join(grid, by = "candidate") |>
  arrange(desc(mean_macro_f1), sd_macro_f1, candidate)
best <- summary_scores[1, ]
oof <- bind_rows(predictions) |> filter(candidate == best$candidate) |> arrange(row)
stopifnot(nrow(oof) == nrow(training_model_data), !anyDuplicated(oof$row))
cm <- as.data.frame(table(Observed = factor(oof$obs, levels = lev),
                         Predicted = factor(oof$pred, levels = lev)))
class_scores <- bind_rows(lapply(lev, function(label) {
  tp <- sum(oof$obs == label & oof$pred == label)
  actual <- sum(oof$obs == label)
  predicted <- sum(oof$pred == label)
  data.frame(class = label, support = actual,
    precision = if (predicted == 0) 0 else tp / predicted,
    recall = tp / actual, f1 = 2 * tp / (actual + predicted))
}))
write_csv(fold_scores, file.path(out, "cv_fold_metrics.csv"))
write_csv(summary_scores, file.path(out, "cv_summary.csv"))
write_csv(bind_rows(baseline), file.path(out, "baseline_cv.csv"))
write_csv(oof, file.path(out, "selected_oof_predictions.csv"))
write_csv(cm, file.path(out, "confusion_matrix.csv"))
write_csv(class_scores, file.path(out, "per_class_metrics.csv"))
write_csv(bind_rows(removed), file.path(out, "removed_predictors.csv"))
saveRDS(list(training_ids = training_data$service_id, test_ids = test_data$service_id,
             cv_index = cv_index, cv_index_out = cv_index_out),
        file.path(out, "shared_split_manifest.rds"))
cat("Refitting chosen configuration on all training observations.\n")
final_recipe <- prep(tree_recipe, training = training_model_data)
final_model <- fit_rf(juice(final_recipe), best, 7503L, importance = "permutation")
saveRDS(list(model = final_model, recipe = final_recipe, configuration = best,
             outcome_levels = lev, test_evaluated = FALSE,
             shared_setup_md5 = tools::md5sum("R/06_model_setup.R")),
        file.path(out, "random_forest_fit.rds"))
importance <- data.frame(feature = names(final_model$variable.importance),
                         importance = unname(final_model$variable.importance)) |>
  arrange(desc(importance))
write_csv(importance, file.path(out, "permutation_importance.csv"))
p <- ggplot(cm, aes(x = Predicted, y = Observed, fill = Freq)) +
  geom_tile() + geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "white", high = "#60b6a8") +
  labs(title = "Random forest: pooled out-of-fold predictions", fill = "Count") +
  theme_minimal(base_size = 12)
ggsave("figures/random_forest/confusion_matrix.png", p, width = 8, height = 5, dpi = 150)
top <- head(importance, 15)
p <- ggplot(top, aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "#268477") + coord_flip() + theme_minimal(base_size = 11) +
  labs(title = "Training-set OOB permutation importance", x = NULL,
       y = "Permutation importance (not causal)")
ggsave("figures/random_forest/importance.png", p, width = 9, height = 6, dpi = 150)
capture.output(sessionInfo(), file = file.path(out, "session_info.txt"))
print(summary_scores)
print(class_scores)
cat("Finished. No held-out test performance was calculated.\n")
