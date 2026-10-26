# Run from the repository root. Reuses the original MLR method and grid,
# but creates separate manifests from the group's actual shared workflow.
source("R/06_model_setup.R")
source("mlr/preprocessing.R")
source("mlr/tuning.R")
source("mlr/interpretation.R")
stopifnot(requireNamespace("glmnet", quietly = TRUE))
out <- "outputs/mlr_shared"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
df <- as.data.frame(model_data_raw)
manifest <- data.frame(service_id = df$service_id, provider_id = df$provider_id,
                       rating_tier = df$rating_tier,
                       partition = ifelse(df$service_id %in% training_data$service_id,
                                          "train", "test"))
fold <- rep(NA_character_, nrow(training_data))
for (f in names(cv_index_out)) fold[cv_index_out[[f]]] <- f
stopifnot(!anyNA(fold))
fold_manifest <- data.frame(service_id = training_data$service_id,
                            provider_id = training_data$provider_id, fold = fold)
write.csv(manifest, file.path(out, "split_manifest.csv"), row.names = FALSE)
write.csv(fold_manifest, file.path(out, "cv_fold_manifest.csv"), row.names = FALSE)
m <- mlr_load_manifests(df, out)
stopifnot(identical(as.character(df$service_id[m$train_rows]),
                    as.character(training_data$service_id)), identical(m$fold, fold))
old <- read.csv("outputs/mlr/split_manifest.csv")
audit <- data.frame(
  check = c("shared_training_rows", "shared_test_rows", "shared_provider_overlap",
            "corrected_training_rows_in_shared_test", "original_mlr_training_rows_in_shared_test"),
  value = c(nrow(training_data), nrow(test_data),
            length(intersect(training_data$provider_id, test_data$provider_id)),
            sum(df$service_id[m$train_rows] %in% test_data$service_id),
            sum(old$service_id[old$partition == "train"] %in% test_data$service_id)))
write.csv(audit, file.path(out, "split_audit.csv"), row.names = FALSE)
set.seed(5003)
mlr_run_tuning(df, manifest_dir = out, out_dir = out)
mlr_fit_selected_train(df, manifest_dir = out, out_dir = out)
capture.output(sessionInfo(), file = file.path(out, "session_info.txt"))
inputs <- c("R/06_model_setup.R", "mlr/preprocessing.R", "mlr/tuning.R",
            "mlr/interpretation.R", "data/processed/acecqa_analysis.csv")
write.csv(data.frame(path = inputs, md5 = unname(tools::md5sum(inputs))),
          file.path(out, "input_hashes.csv"), row.names = FALSE)
