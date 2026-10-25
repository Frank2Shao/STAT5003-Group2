# MLR tuning on the EXACT shared provider-grouped folds.
# Uses recipe from mlr/preprocessing.R (prepared only on each analysis fold).
# F1 for an unpredicted class is zero; all three classes are always included.

mlr_class_levels <- c("Below NQS", "Meeting NQS", "Above NQS")

mlr_fold_scores <- function(truth, pred) {
  truth <- factor(truth, levels = mlr_class_levels)
  pred <- factor(pred, levels = mlr_class_levels)
  stopifnot(!anyNA(truth), !anyNA(pred), length(truth) > 0L)
  m <- table(truth, pred)
  tp <- diag(m)
  p <- ifelse(colSums(m) > 0, tp / pmax(colSums(m), 1), 0)
  r <- ifelse(rowSums(m) > 0, tp / pmax(rowSums(m), 1), 0)
  f1 <- ifelse(p + r > 0, 2 * p * r / pmax(p + r, .Machine$double.eps), 0)
  list(macro_f1 = mean(f1), accuracy = sum(tp) / sum(m),
       per_class = data.frame(class = mlr_class_levels, precision = p,
                              recall = r, f1 = f1, support = rowSums(m)),
       confusion = m)
}

mlr_training_weights <- function(y, scheme = c("none", "sqrt_inverse", "inverse_capped")) {
  scheme <- match.arg(scheme)
  counts <- table(factor(y, levels = mlr_class_levels))
  if (any(counts == 0)) stop("Each training fold must contain every class")
  if (scheme == "none") return(rep(1, length(y)))
  raw <- as.numeric(mean(counts) / counts)
  names(raw) <- names(counts)
  if (scheme == "sqrt_inverse") raw <- sqrt(raw)
  if (scheme == "inverse_capped") raw <- pmin(raw, 5)
  w <- unname(raw[as.character(y)])
  w / mean(w)
}

mlr_load_manifests <- function(df, manifest_dir) {
  a <- utils::read.csv(file.path(manifest_dir, "split_manifest.csv"),
                       colClasses = "character", check.names = FALSE)
  b <- utils::read.csv(file.path(manifest_dir, "cv_fold_manifest.csv"),
                       colClasses = "character", check.names = FALSE)
  if (anyDuplicated(df$service_id) || anyDuplicated(a$service_id) ||
      anyDuplicated(b$service_id)) stop("Duplicate service_id in data or manifests")
  if (!setequal(as.character(df$service_id), a$service_id))
    stop("Split manifest does not match the supplied dataset")
  a <- a[match(as.character(df$service_id), a$service_id), ]
  if (!all(a$partition %in% c("train", "test"))) stop("Unknown partition")
  if (!identical(as.character(df$provider_id), a$provider_id))
    stop("Provider IDs differ between data and split manifest")
  if (anyDuplicated(a$provider_id[a$partition == "train"]) == 0L) {
    # This is fine: providers may have one service.
  }
  if (length(intersect(unique(a$provider_id[a$partition == "train"]),
                       unique(a$provider_id[a$partition == "test"]))))
    stop("Provider leakage between train and test")
  tr <- which(a$partition == "train")
  if (!setequal(as.character(df$service_id[tr]), b$service_id))
    stop("CV manifest does not match training rows")
  b <- b[match(as.character(df$service_id[tr]), b$service_id), ]
  if (!identical(as.character(df$provider_id[tr]), b$provider_id))
    stop("Provider IDs differ between data and CV manifest")
  if (anyNA(b$fold) || length(unique(b$fold)) != 5L)
    stop("Expected exactly five nonmissing CV folds")
  if (any(tapply(b$fold, b$provider_id, function(z) length(unique(z))) != 1L))
    stop("Provider appears in multiple CV folds")
  list(train_rows = tr, fold = b$fold, fold_ids = sort(unique(b$fold)))
}

mlr_run_tuning <- function(df, manifest_dir, out_dir,
                           lambda_grid = c(0.001, 0.003, 0.01, 0.03, 0.1, 0.3),
                           weight_schemes = c("none", "sqrt_inverse", "inverse_capped")) {
  stopifnot(all(lambda_grid > 0), !anyDuplicated(lambda_grid),
            all(weight_schemes %in% c("none", "sqrt_inverse", "inverse_capped")))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  manifests <- mlr_load_manifests(df, manifest_dir)
  train <- mlr_model_data(df[manifests$train_rows, , drop = FALSE])
  fold_metrics <- list()
  class_metrics <- list()
  all_predictions <- list()
  k <- 0L
  for (fold_id in manifests$fold_ids) {
    idx_train <- which(manifests$fold != fold_id)
    idx_valid <- which(manifests$fold == fold_id)
    training <- train[idx_train, , drop = FALSE]
    validation <- train[idx_valid, , drop = FALSE]
    # Crucial: estimate preprocessing parameters ONLY from this fold's training rows.
    rec <- recipes::prep(make_mlr_recipe(training), training = training, retain = TRUE)
    x_train <- recipes::bake(rec, new_data = training, recipes::all_predictors())
    x_valid <- recipes::bake(rec, new_data = validation, recipes::all_predictors())
    if (!identical(names(x_train), names(x_valid))) stop("Recipe column mismatch")
    x_train <- as.matrix(x_train); x_valid <- as.matrix(x_valid)
    storage.mode(x_train) <- "double"; storage.mode(x_valid) <- "double"
    y_train <- factor(training$rating_tier, levels = mlr_class_levels)
    for (scheme in weight_schemes) {
      message("CV ", fold_id, " | weights=", scheme)
      w <- mlr_training_weights(y_train, scheme)
      # Fit all lambdas in a single glmnet path. 'grouped' multinomial coefficient
      # penalties are not required: use default ungrouped ridge, alpha=0.
      fit <- glmnet::glmnet(
        x = x_train, y = y_train, family = "multinomial", alpha = 0,
        lambda = sort(lambda_grid, decreasing = TRUE), weights = w,
        standardize = FALSE
      )
      for (lam in lambda_grid) {
        y_hat <- as.character(predict(fit, newx = x_valid, s = lam, type = "class"))
        sc <- mlr_fold_scores(validation$rating_tier, y_hat)
        k <- k + 1L
        fold_metrics[[k]] <- data.frame(fold = fold_id, weights = scheme,
                                        lambda = lam, macro_f1 = sc$macro_f1,
                                        accuracy = sc$accuracy,
                                        n_valid = length(idx_valid))
        class_metrics[[k]] <- cbind(data.frame(fold = fold_id, weights = scheme,
                                               lambda = lam), sc$per_class)
        # Only save full OOF prediction records for each configuration; useful for auditing.
        all_predictions[[k]] <- data.frame(
          fold = fold_id, weights = scheme, lambda = lam,
          service_id = as.character(df$service_id[manifests$train_rows[idx_valid]]),
          rating_tier = as.character(validation$rating_tier), .pred_class = y_hat)
      }
    }
  }
  fm <- do.call(rbind, fold_metrics)
  pc <- do.call(rbind, class_metrics)
  pp <- do.call(rbind, all_predictions)
  key <- interaction(fm$weights, fm$lambda, drop = TRUE)
  summary <- do.call(rbind, lapply(split(fm, key), function(z) {
    data.frame(weights = z$weights[1], lambda = z$lambda[1],
               mean_macro_f1 = mean(z$macro_f1), sd_macro_f1 = stats::sd(z$macro_f1),
               mean_accuracy = mean(z$accuracy), folds = nrow(z))
  }))
  summary <- summary[order(-summary$mean_macro_f1, summary$sd_macro_f1,
                           summary$lambda), , drop = FALSE]
  rownames(summary) <- NULL
  if (any(summary$folds != length(manifests$fold_ids))) stop("Incomplete CV grid")
  best <- summary[1, , drop = FALSE]
  utils::write.csv(fm, file.path(out_dir, "mlr_tuning_fold_metrics.csv"), row.names = FALSE)
  utils::write.csv(pc, file.path(out_dir, "mlr_tuning_class_metrics.csv"), row.names = FALSE)
  utils::write.csv(pp, file.path(out_dir, "mlr_tuning_oof_predictions.csv"), row.names = FALSE)
  utils::write.csv(summary, file.path(out_dir, "mlr_tuning_summary.csv"), row.names = FALSE)
  utils::write.csv(best, file.path(out_dir, "mlr_tuning_best_params.csv"), row.names = FALSE)
  message("Best CV config (not test-set performance):")
  print(best)
  message("No test set rows used in fitting or tuning.")
  invisible(list(summary = summary, best = best))
}
