required_packages <- c(
  "readr",
  "dplyr",
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
dir.create("outputs/lda/", recursive=TRUE, showWarnings=FALSE)

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
  tuneGrid = expand.grid(dimen=1:2), # Only 2 possible values for hyperparam tuning
  prior = rep(1/3, 3) # To account for class imbalance in the dataset 
)

lda_fit_emp <- train(
  lda_recipe,
  data = training_model_data,
  trControl = model_control,
  method = "lda2",
  metric = "Macro_F1",
  tuneGrid = expand.grid(dimen=1:2) # Only 2 possible values for hyperparam tuning
)

# Performance on data
## With vs. Without Class Priors
### Side by side
lda_prior_comparison <- bind_rows(
  lda_fit_emp$results |> mutate(priors = "Empirical"),
  lda_fit_equal$results   |> mutate(priors = "Equal")
) %>%
  select(priors, dimen, Macro_F1, Accuracy, starts_with("Recall_"))

print(lda_prior_comparison)

### Per-fold comparison
lda_fold_comparison <- bind_rows(
  lda_fit_emp$resample |> mutate(priors = "Empirical"),
  lda_fit_equal$resample   |> mutate(priors = "Equal")
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

## Test Data
lda.preds<- predict(lda_fit_equal, newdata=test_model_data)
lda.confusion <- confusionMatrix(data=lda.preds, reference=test_model_data$rating_tier)
lda.confusion

## Training Data (Out of Fold)
lda.train.confusion <- confusionMatrix(data=lda_fit_equal$pred$pred, reference=lda_fit_equal$pred$obs)
lda.train.confusion

# Evaluations

lda_tier_levels <- levels(test_model_data$rating_tier)

## Out-of-fold predictions for the chosen dimen
lda_oof <- lda_fit_equal$pred
if ("dimen" %in% names(lda_oof)) {
  lda_oof <- lda_oof %>% filter(dimen == lda_fit_equal$bestTune$dimen)
}
stopifnot(nrow(lda_oof) == nrow(training_model_data))   # one prediction per service

lda_oof_df  <- data.frame(obs = lda_oof$obs, pred = lda_oof$pred)
lda_test_df <- data.frame(obs = test_model_data$rating_tier, pred = lda.preds)

## Overall metrics: CV (fold mean), CV (pooled OOF), test
lda_best_row <- lda_fit_equal$results %>% filter(dimen == lda_fit_equal$bestTune$dimen)

lda_overall <- bind_rows(
  data.frame(split = "CV (mean of folds)",
             t(unlist(lda_best_row[, c("Macro_F1", "Accuracy")]))),
  data.frame(split = "CV (pooled out-of-fold)",
             t(macro_f1_summary(lda_oof_df,  lev = lda_tier_levels)[c("Macro_F1", "Accuracy")])),
  data.frame(split = "Test",
             t(macro_f1_summary(lda_test_df, lev = lda_tier_levels)[c("Macro_F1", "Accuracy")]))
)
print(lda_overall)

## Majority-class baseline for reference
majority <- names(which.max(table(training_model_data$rating_tier)))
baseline <- data.frame(
  obs  = test_model_data$rating_tier,
  pred = factor(majority, levels = lda_tier_levels)[rep(1, nrow(test_model_data))]
)
print(macro_f1_summary(baseline, lev = lda_tier_levels)[c("Macro_F1", "Accuracy")])

## Per-class precision / recall / F1 
lda_per_class <- function(cm, split) {
  bc <- as.data.frame(cm$byClass)
  bc$class <- sub("^Class: ", "", rownames(bc))
  bc$split <- split
  bc[, c("split", "class", "Precision", "Recall", "F1")]
}

lda_class_metrics <- bind_rows(
  lda_per_class(lda.train.confusion, "CV (out-of-fold)"),
  lda_per_class(lda.confusion,       "Test")
)
print(lda_class_metrics)
write_csv(lda_class_metrics, "outputs/lda/LDA_class_metrics.csv")

## Ordinal view: quadratic weighted kappa and how far off errors are
lda_qw_kappa <- function(obs, pred) {
  k <- nlevels(obs)
  O <- table(obs, pred); O <- O / sum(O)
  E <- outer(rowSums(O), colSums(O))
  W <- outer(seq_len(k), seq_len(k), function(i, j) (i - j)^2 / (k - 1)^2)
  1 - sum(W * O) / sum(W * E)
}

cat("\nQuadratic weighted kappa  CV:  ", round(lda_qw_kappa(lda_oof_df$obs,  lda_oof_df$pred),  2),
    "\nQuadratic weighted kappa  Test:", round(lda_qw_kappa(lda_test_df$obs, lda_test_df$pred), 2), "\n")

lda_tier_distance <- abs(as.integer(lda_test_df$obs) - as.integer(lda_test_df$pred))
cat("\nTest error distance (0 = correct, 1 = adjacent tier, 2 = Below<->Above):\n")
print(round(100 * prop.table(table(lda_tier_distance)), 1))

## Provider-level bootstrap CI for test Macro F1 
stopifnot(nrow(test_data) == nrow(test_model_data))

set.seed(5005)
provider_rows <- split(seq_len(nrow(test_data)), test_data$provider_id)
providers     <- names(provider_rows)

lda_boot_f1 <- replicate(1000, {
  sampled <- sample(providers, replace = TRUE)
  idx     <- unlist(provider_rows[sampled], use.names = FALSE)
  macro_f1_summary(lda_test_df[idx, ], lev = lda_tier_levels)[["Macro_F1"]]
})

cat("\nTest Macro F1 95% bootstrap CI (resampling providers):",
    round(quantile(lda_boot_f1, c(0.025, 0.975)), 3), "\n")

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

lda.test.plot  <- lda_plot_cm(lda.confusion,       "LDA Predictions on Test Set")
lda.train.plot <- lda_plot_cm(lda.train.confusion, "LDA Cross-Validated Predictions (Training Set)")

lda.metric.plot <- lda_class_metrics %>%
  tidyr::pivot_longer(c(Precision, Recall, F1), names_to = "metric", values_to = "value") %>%
  mutate(class = factor(class, levels = lda_tier_levels)) %>%
  ggplot(aes(class, value, fill = split)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ metric) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "LDA per-class metrics", x = NULL, y = NULL, fill = NULL) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave("figures/lda_train_cm.png",      lda.train.plot,  width = 6, height = 6, dpi = 220)
ggsave("figures/lda_test_cm.png",       lda.test.plot,   width = 6, height = 6, dpi = 220)
ggsave("figures/lda_class_metrics.png", lda.metric.plot, width = 9, height = 4, dpi = 220)