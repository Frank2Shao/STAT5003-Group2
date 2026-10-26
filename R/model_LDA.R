required_packages <- c(
  "readr",
  "dplyr",
  "tidyr",
  "caret",
  "recipes",
  "ggplot2"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0) {
  stop(
    "Install the following packages before running this script: ",
    paste(missing_packages, collapse = ", ")
  )
}

library(readr)
library(dplyr)
library(caret)
library(recipes)
library(ggplot2)
select <- dplyr::select

if (!exists("scaled_recipe")) {source("R/06_model_setup.R")}
dir.create("outputs/lda", recursive = TRUE, showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

# Training Model

lda_recipe <- scaled_recipe %>% # Additional pre-processing
  step_lincomb(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.9)

lda_fit_equal <- train(
  lda_recipe,
  data = training_model_data,
  trControl = model_control,
  method = "lda2",
  metric = "Macro_F1",
  tuneGrid = expand.grid(dimen = 1:2), # Only 2 possible values for hyperparam tuning
  prior = rep(1/3, 3) # Equal class priors
)

lda_fit_emp <- train(
  lda_recipe,
  data = training_model_data,
  trControl = model_control,
  method = "lda2",
  metric = "Macro_F1",
  tuneGrid = expand.grid(dimen = 1:2) # Empirical priors (training class proportions)
)

# Performance on data (cross-validation only)
## With vs. Without Class Priors
### Side by side
lda_prior_comparison <- bind_rows(
  lda_fit_emp$results   |> mutate(priors = "Empirical"),
  lda_fit_equal$results |> mutate(priors = "Equal")
) %>%
  select(priors, dimen, Macro_F1, Accuracy, starts_with("Recall_"))

print(lda_prior_comparison)

### Per-fold comparison
lda_fold_comparison <- bind_rows(
  lda_fit_emp$resample   |> mutate(priors = "Empirical"),
  lda_fit_equal$resample |> mutate(priors = "Equal")
)

lda_fold_comparison %>%
  group_by(priors) %>%
  summarise(
    mean_macro_f1 = mean(Macro_F1),
    sd_macro_f1   = sd(Macro_F1),
    .groups = "drop"
  )

### Paired difference per fold (equal - emp)
lda_paired_diff <- lda_fold_comparison %>%
  select(Resample, priors, Macro_F1) %>%
  tidyr::pivot_wider(names_from = priors, values_from = Macro_F1) %>%
  mutate(diff = `Equal` - `Empirical`)
print(lda_paired_diff)

## Out-of-fold predictions for the chosen dimen
lda_tier_levels <- levels(training_model_data$rating_tier)

lda_oof <- lda_fit_equal$pred
if ("dimen" %in% names(lda_oof)) {
  lda_oof <- lda_oof %>% filter(dimen == lda_fit_equal$bestTune$dimen)
}
stopifnot(nrow(lda_oof) == nrow(training_model_data))   # one prediction per service

lda_oof_df <- data.frame(obs = lda_oof$obs, pred = lda_oof$pred)

lda.train.confusion <- confusionMatrix(data = lda_oof_df$pred, reference = lda_oof_df$obs)
lda.train.confusion

# Evaluations (cross-validation only)

## Overall metrics: CV (fold mean) and CV (pooled out-of-fold)
lda_best_row <- lda_fit_equal$results %>% filter(dimen == lda_fit_equal$bestTune$dimen)

lda_overall <- bind_rows(
  data.frame(split = "CV (mean of folds)",
             t(unlist(lda_best_row[, c("Macro_F1", "Accuracy")]))),
  data.frame(split = "CV (pooled out-of-fold)",
             t(macro_f1_summary(lda_oof_df, lev = lda_tier_levels)[c("Macro_F1", "Accuracy")]))
)
print(lda_overall)

## Per-class precision / recall / F1
lda_per_class <- function(cm, split) {
  bc <- as.data.frame(cm$byClass)
  bc$class <- sub("^Class: ", "", rownames(bc))
  bc$split <- split
  bc[, c("split", "class", "Precision", "Recall", "F1")]
}

lda_class_metrics <- lda_per_class(lda.train.confusion, "CV (out-of-fold)")
print(lda_class_metrics)

## Baseline vs LDA (majority class per fold, scored on the same validation folds)
fold_baseline <- function(fold) {
  majority <- names(which.max(table(training_model_data$rating_tier[cv_index[[fold]]])))
  rows <- cv_index_out[[fold]]
  d <- data.frame(
    obs  = training_model_data$rating_tier[rows],
    pred = factor(rep(majority, length(rows)), levels = lda_tier_levels)
  )
  as.data.frame(t(macro_f1_summary(d, lev = lda_tier_levels)))
}

lda_vs_baseline <- bind_rows(
  `Majority baseline`  = colMeans(bind_rows(lapply(names(cv_index_out), fold_baseline))),
  `LDA (empirical)`    = unlist(lda_fit_emp$results[lda_fit_emp$results$dimen == lda_fit_emp$bestTune$dimen, ]),
  `LDA (equal priors)` = unlist(lda_fit_equal$results[lda_fit_equal$results$dimen == lda_fit_equal$bestTune$dimen, ]),
  .id = "model"
) |>
  dplyr::select(model, Macro_F1, Accuracy, starts_with("Recall_")) |>
  tidyr::pivot_longer(-model, names_to = "metric") |>
  tidyr::pivot_wider(names_from = model)

print(lda_vs_baseline)

## Plots
lda_plot_cm <- function(cm, title) {
  df <- as.data.frame(cm$table) %>%
    group_by(Reference) %>%
    mutate(row_pct = 100 * Freq / sum(Freq)) %>%
    ungroup()
  
  ggplot(df, aes(Prediction, Reference, fill = row_pct)) +
    geom_tile(colour = "white") +
    geom_text(aes(label = paste0(Freq, "\n(", round(row_pct, 1), "%)")), size = 4) +
    scale_fill_gradient(low = "white", high = "steelblue", name = "% of true class") +
    scale_y_discrete(limits = rev(levels(df$Reference))) +
    labs(title = title, x = "Predicted rating", y = "Actual rating") +
    coord_fixed() +
    theme_minimal()
}

lda.train.plot <- lda_plot_cm(lda.train.confusion, "LDA Cross-Validated Predictions")

lda.metric.plot <- lda_class_metrics %>%
  tidyr::pivot_longer(c(Precision, Recall, F1), names_to = "metric", values_to = "value") %>%
  mutate(class = factor(class, levels = lda_tier_levels)) %>%
  ggplot(aes(class, value)) +
  geom_col(fill = "steelblue", width = 0.7) +
  facet_wrap(~ metric) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "LDA per-class metrics", x = NULL, y = NULL) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# Saved outputs
write_csv(lda_class_metrics,         "outputs/lda/lda_class_metrics.csv")
write_csv(lda_fit_equal$resample,    "outputs/lda/lda_cv_fold_metrics.csv")
ggsave("figures/lda_train_cm.png",      lda.train.plot,  width = 6, height = 6, dpi = 220)
ggsave("figures/lda_class_metrics.png", lda.metric.plot, width = 6, height = 4, dpi = 220)

message("Completed.")