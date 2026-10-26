# Run after R/11_mlr_shared_split.R, from the repository root.
# Freeze the CV choice before computing any test metrics. Never tune on test scores.
source("R/06_model_setup.R")
source("mlr/preprocessing.R")
source("mlr/tuning.R")
library(glmnet)
out <- "outputs/selected_model"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
dir.create("figures/selected_model", recursive = TRUE, showWarnings = FALSE)
read <- function(path) read.csv(path, check.names = FALSE)
save_csv <- function(x, name) write.csv(x, file.path(out, name), row.names = FALSE)
cfg <- read("outputs/mlr_shared/mlr_selected_config.csv")
mlr <- read("outputs/mlr_shared/mlr_tuning_fold_metrics.csv")
mlr <- mlr[mlr$weights == cfg$weights & abs(mlr$lambda - cfg$lambda) < 1e-12, ]
rf_cfg <- read("outputs/random_forest/cv_summary.csv")[1, ]
rf <- read("outputs/random_forest/cv_fold_metrics.csv")
rf <- rf[rf$candidate == rf_cfg$candidate, ]
knn_cfg <- read(file.path(out, "reference/knn_best_parameters.csv"))
knn <- read(file.path(out, "reference/knn_cv_fold_metrics.csv"))
knn <- knn[knn$k == knn_cfg$k & knn$kernel == knn_cfg$kernel, ]
lda <- read(file.path(out, "reference/lda_cv_fold_metrics.csv"))
svm <- read("outputs/models/svm/svm_rbf_selected_resample_results.csv")
base <- read("outputs/random_forest/baseline_cv.csv")
harmonise <- function(model, fold, f1, accuracy) {
  stopifnot(length(fold) == 5L, setequal(fold, paste0("Fold", 1:5)),
            !anyDuplicated(fold), all(is.finite(f1)), all(is.finite(accuracy)))
  data.frame(model = model, fold = fold, macro_f1 = f1, accuracy = accuracy)
}
folds <- rbind(
  harmonise("MLR (shared split)", mlr$fold, mlr$macro_f1, mlr$accuracy),
  harmonise("Random Forest", rf$fold, rf$Macro_F1, rf$Accuracy),
  harmonise("SVM (RBF)", svm$Resample, svm$Macro_F1, svm$Accuracy),
  harmonise("LDA", lda$Resample, lda$Macro_F1, lda$Accuracy),
  harmonise("KNN", knn$fold, knn$Macro_F1, knn$Accuracy),
  harmonise("Majority baseline", base$fold, base$Macro_F1, base$Accuracy))
ranking <- do.call(rbind, lapply(split(folds, folds$model), function(z) {
  data.frame(model = z$model[1], mean_macro_f1 = mean(z$macro_f1),
             sd_macro_f1 = sd(z$macro_f1), mean_accuracy = mean(z$accuracy))
}))
ranking <- ranking[order(-ranking$mean_macro_f1), ]
rownames(ranking) <- NULL
stopifnot(ranking$model[1] == "MLR (shared split)")
save_csv(folds, "comparison_fold_metrics.csv")
save_csv(ranking, "comparison_summary.csv")
print(ranking)

df <- as.data.frame(model_data_raw)
manifest <- mlr_load_manifests(df, "outputs/mlr_shared")
stopifnot(identical(as.character(df$service_id[manifest$train_rows]),
                    as.character(training_data$service_id)))
for (f in names(cv_index_out)) stopifnot(all(manifest$fold[cv_index_out[[f]]] == f))
train <- mlr_model_data(training_data)
test <- mlr_model_data(test_data)
fit <- readRDS("outputs/mlr_shared/mlr_selected_train_model.rds")
stopifnot(fit$selected_config$lambda == cfg$lambda,
          fit$selected_config$weights == cfg$weights,
          nrow(fit$recipe$template) == nrow(train),
          length(intersect(training_data$provider_id, test_data$provider_id)) == 0)
# Re-prepping verifies that the saved preprocessing was learned on these training rows.
check_recipe <- recipes::prep(make_mlr_recipe(train), training = train, retain = TRUE)
x_train <- as.matrix(recipes::bake(fit$recipe, new_data = train, recipes::all_predictors()))
x_check <- as.matrix(recipes::bake(check_recipe, new_data = train, recipes::all_predictors()))
stopifnot(isTRUE(all.equal(x_train, x_check, tolerance = 1e-12)))
x_test <- as.matrix(recipes::bake(fit$recipe, new_data = test, recipes::all_predictors()))
storage.mode(x_test) <- "double"
stopifnot(identical(colnames(x_test), fit$model_feature_columns), !anyNA(x_test))
inputs <- c("outputs/mlr_shared/mlr_selected_train_model.rds",
            "outputs/mlr_shared/split_manifest.csv", "outputs/mlr_shared/cv_fold_manifest.csv",
            "outputs/mlr_shared/mlr_selected_config.csv", "data/processed/acecqa_analysis.csv")
save_csv(data.frame(path = inputs, md5 = unname(tools::md5sum(inputs))), "input_hashes.csv")
save_csv(data.frame(model = ranking$model[1], lambda = cfg$lambda, weights = cfg$weights,
                    criterion = "highest mean training CV Macro-F1",
                    selected_before_test_scoring = TRUE), "selection.csv")

pred <- as.character(predict(fit$model, newx = x_test, s = cfg$lambda, type = "class"))
score <- mlr_fold_scores(test$rating_tier, pred)
majority <- names(which.max(table(train$rating_tier)))
baseline <- mlr_fold_scores(test$rating_tier, rep(majority, nrow(test)))
metrics <- data.frame(model = c("MLR (shared split)", "Majority baseline"),
                       macro_f1 = c(score$macro_f1, baseline$macro_f1),
                       accuracy = c(score$accuracy, baseline$accuracy),
                       n_test = nrow(test))
save_csv(metrics, "test_metrics.csv")
save_csv(score$per_class, "test_class_metrics.csv")
cm <- as.data.frame(score$confusion)
names(cm) <- c("Observed", "Predicted", "Count")
save_csv(cm, "test_confusion_matrix.csv")
save_csv(data.frame(service_id = test_data$service_id, provider_id = test_data$provider_id,
                    observed = test$rating_tier, predicted = pred), "test_predictions.csv")
save_csv(data.frame(check = c("training_rows", "test_rows", "training_providers", "test_providers",
                              "overlapping_providers", "preprocessing_matches_training_only"),
                    value = c(nrow(train), nrow(test), length(unique(training_data$provider_id)),
                              length(unique(test_data$provider_id)), 0, 1)), "evaluation_audit.csv")
p <- ggplot(cm, aes(Predicted, Observed, fill = Count)) +
  geom_tile() + geom_text(aes(label = Count), size = 5) +
  scale_fill_gradient(low = "white", high = "#71a499") +
  scale_y_discrete(limits = rev(mlr_class_levels)) +
  labs(x = "Predicted tier", y = "Actual tier") + theme_minimal(base_size = 14) +
  theme(panel.grid = element_blank(), legend.position = "none")
ggsave("figures/selected_model/mlr_test_confusion.png", p, width = 7, height = 4.8, dpi = 180)
capture.output(sessionInfo(), file = file.path(out, "session_info.txt"))
print(metrics)
print(score$per_class)
cat("This corrected model excludes the shared test set from fitting and tuning.\n",
    "The project's test partition was previously exposed in other model work;\n",
    "these results are NOT a pristine independent final confirmation.\n")
