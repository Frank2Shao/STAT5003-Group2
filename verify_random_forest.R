library(readr)
library(dplyr)
library(ranger)
folder <- "outputs/random_forest"
scores <- read_csv(file.path(folder, "cv_fold_metrics.csv"), show_col_types = FALSE)
ranking <- read_csv(file.path(folder, "cv_summary.csv"), show_col_types = FALSE)
oof <- read_csv(file.path(folder, "selected_oof_predictions.csv"), show_col_types = FALSE)
manifest <- readRDS(file.path(folder, "shared_split_manifest.rds"))
bundle <- readRDS(file.path(folder, "random_forest_fit.rds"))
stopifnot(nrow(scores) == 40L, nrow(ranking) == 8L,
          all(table(scores$candidate) == 5L),
          all(is.finite(scores$Macro_F1)), all(scores$Macro_F1 >= 0),
          all(scores$Macro_F1 <= 1),
          nrow(oof) == length(manifest$training_ids),
          !anyDuplicated(oof$service_id),
          identical(oof$service_id, manifest$training_ids),
          !any(oof$service_id %in% manifest$test_ids),
          all(oof$obs %in% bundle$outcome_levels),
          all(oof$pred %in% bundle$outcome_levels),
          identical(bundle$test_evaluated, FALSE))
for (fold in names(manifest$cv_index)) {
  a <- manifest$cv_index[[fold]]
  b <- manifest$cv_index_out[[fold]]
  stopifnot(!any(oof$provider_id[a] %in% oof$provider_id[b]),
            all(oof$fold[b] == fold))
}
for (i in ranking$candidate) {
  expected <- mean(scores$Macro_F1[scores$candidate == i])
  stopifnot(abs(expected - ranking$mean_macro_f1[ranking$candidate == i]) < 1e-12)
}
class_scores <- read_csv(file.path(folder, "per_class_metrics.csv"), show_col_types = FALSE)
for (label in bundle$outcome_levels) {
  tp <- sum(oof$obs == label & oof$pred == label)
  expected <- 2 * tp / (sum(oof$obs == label) + sum(oof$pred == label))
  stopifnot(abs(class_scores$f1[class_scores$class == label] - expected) < 1e-12)
}
# Check that the saved recipe/model can be reloaded and used together.
raw <- read_csv("data/processed/acecqa_analysis.csv", show_col_types = FALSE)
dictionary <- read_csv("data/processed/acecqa_variable_dictionary.csv", show_col_types = FALSE)
sample <- raw[match(head(manifest$training_ids, 20), raw$service_id), ]
logical_names <- dictionary$variable[dictionary$type == "Logical"]
for (name in logical_names) sample[[name]] <- factor(sample[[name]],
  levels = c(FALSE, TRUE), labels = c("No", "Yes"))
sample$rating_tier <- factor(sample$rating_tier, levels = bundle$outcome_levels)
baked <- recipes::bake(bundle$recipe, new_data = sample)
pred <- predict(bundle$model, data = as.data.frame(baked))$predictions
stopifnot(length(pred) == 20L, !anyNA(pred))
cat("PASS: 40 CV fits, ranking, OOF coverage, provider isolation, metrics, saved-model prediction.\n")
