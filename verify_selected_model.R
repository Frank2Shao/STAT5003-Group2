# Verify saved evidence without fitting models or making new test predictions.
source("mlr/tuning.R")
read <- function(path) read.csv(path, check.names = FALSE)
o <- "outputs/selected_model/"
pred <- read(paste0(o, "test_predictions.csv"))
manifest <- read("outputs/mlr_shared/split_manifest.csv")
folds <- read("outputs/mlr_shared/cv_fold_manifest.csv")
stopifnot(nrow(pred) == 3663L, !anyDuplicated(pred$service_id),
          setequal(pred$service_id, manifest$service_id[manifest$partition == "test"]),
          setequal(folds$service_id, manifest$service_id[manifest$partition == "train"]),
          length(intersect(manifest$provider_id[manifest$partition == "train"],
                           manifest$provider_id[manifest$partition == "test"])) == 0,
          all(tapply(folds$fold, folds$provider_id, function(x) length(unique(x))) == 1L))
sc <- mlr_fold_scores(pred$observed, pred$predicted)
metrics <- read(paste0(o, "test_metrics.csv"))
stopifnot(abs(sc$macro_f1 - metrics$macro_f1[1]) < 1e-12,
          abs(sc$accuracy - metrics$accuracy[1]) < 1e-12)
cm <- as.data.frame(sc$confusion)
saved_cm <- read(paste0(o, "test_confusion_matrix.csv"))
stopifnot(identical(as.integer(cm$Freq), as.integer(saved_cm$Count)),
          identical(as.character(cm$truth), saved_cm$Observed),
          identical(as.character(cm$pred), saved_cm$Predicted))
pc <- read(paste0(o, "test_class_metrics.csv"))
for (column in c("precision", "recall", "f1", "support")) {
  stopifnot(max(abs(pc[[column]] - sc$per_class[[column]])) < 1e-12)
}
majority <- names(which.max(table(manifest$rating_tier[manifest$partition == "train"])))
base <- mlr_fold_scores(pred$observed, rep(majority, nrow(pred)))
stopifnot(abs(base$macro_f1 - metrics$macro_f1[2]) < 1e-12,
          abs(base$accuracy - metrics$accuracy[2]) < 1e-12)
cv <- read(paste0(o, "comparison_fold_metrics.csv"))
ranking <- read(paste0(o, "comparison_summary.csv"))
for (i in seq_len(nrow(ranking))) {
  z <- cv[cv$model == ranking$model[i], ]
  stopifnot(nrow(z) == 5L, !anyDuplicated(z$fold),
            abs(mean(z$macro_f1) - ranking$mean_macro_f1[i]) < 1e-12)
}
stopifnot(ranking$model[1] == "MLR (shared split)", all(diff(ranking$mean_macro_f1) <= 0))
hashes <- read(paste0(o, "input_hashes.csv"))
stopifnot(identical(unname(tools::md5sum(hashes$path)), hashes$md5))
cat("PASS: partitions, provider separation, predictions, class metrics, confusion matrix, baseline, CV ranking and input hashes.\n")
