# Interpretation and shareable CV result export for the selected ridge multinomial MLR.
# Only training partition used. This module does NOT evaluate held-out test rows.

mlr_fit_selected_train <- function(df, manifest_dir, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  manifest <- mlr_load_manifests(df, manifest_dir)
  train <- mlr_model_data(df[manifest$train_rows, , drop = FALSE])
  summary_file <- file.path(manifest_dir, 'mlr_tuning_summary.csv')
  if (!file.exists(summary_file)) stop('Missing tuning summary: ', summary_file)
  ranking <- utils::read.csv(summary_file, stringsAsFactors = FALSE)
  must <- c('weights','lambda','mean_macro_f1','sd_macro_f1','mean_accuracy','folds')
  if (!all(must %in% names(ranking))) stop('Unexpected tuning summary columns')
  if (anyNA(ranking[, must])) stop('Incomplete tuning summary')
  ranking <- ranking[order(-ranking$mean_macro_f1, ranking$sd_macro_f1, ranking$lambda), ]
  best <- ranking[1L, , drop = FALSE]
  if (best$folds != 5L) stop('Best configuration must have 5 complete folds')
  if (best$weights %notin% c('none','sqrt_inverse','inverse_capped')) stop('Unknown weight scheme')
  if (!is.finite(best$lambda) || best$lambda <= 0) stop('Invalid lambda')

  # Fit preprocessing on training rows only; no test data touched.
  recipe_fit <- recipes::prep(make_mlr_recipe(train), training = train, retain = TRUE)
  features <- recipes::bake(recipe_fit, new_data = train, recipes::all_predictors())
  x <- as.matrix(features)
  storage.mode(x) <- 'double'
  y <- factor(train$rating_tier, levels = mlr_class_levels)
  w <- mlr_training_weights(y, best$weights)
  fitted <- glmnet::glmnet(x, y, family = 'multinomial', alpha = 0,
                          lambda = best$lambda, weights = w, standardize = FALSE)

  # Coefficients represent log odds relative to the other classes in the
  # multinomial softmax parameterisation; exp(beta) is the multiplicative
  # change in exp(eta_class), not a binary one-vs-reference odds ratio.
  coef_by_class <- stats::coef(fitted, s = best$lambda)
  rows <- lapply(names(coef_by_class), function(class_label) {
    cc <- as.matrix(coef_by_class[[class_label]])
    z <- data.frame(class = class_label, feature = rownames(cc),
                    coefficient = as.numeric(cc[, 1L]), stringsAsFactors = FALSE)
    z$exp_coefficient <- exp(pmax(pmin(z$coefficient, 700), -700))
    z
  })
  coef_table <- do.call(rbind, rows)
  coef_table <- coef_table[order(coef_table$class, -abs(coef_table$coefficient)), ]
  rownames(coef_table) <- NULL
  utils::write.csv(coef_table, file.path(out_dir, 'mlr_selected_coefficients.csv'), row.names = FALSE)
  utils::write.csv(coef_table[coef_table$feature != '(Intercept)', ],
                   file.path(out_dir, 'mlr_selected_feature_coefficients.csv'), row.names = FALSE)

  # Uniform group comparison row: all metrics are CV-only; no holdout estimates.
  model_summary <- data.frame(model = 'MLR (ridge, tuned)',
                              cv_scheme = '5-fold provider-grouped',
                              selection_metric = 'macro_f1',
                              cv_mean_macro_f1 = best$mean_macro_f1,
                              cv_sd_macro_f1 = best$sd_macro_f1,
                              cv_mean_accuracy = best$mean_accuracy,
                              penalty = best$lambda,
                              weighting = best$weights,
                              test_metrics_available = FALSE)
  utils::write.csv(model_summary, file.path(out_dir, 'mlr_group_comparison.csv'), row.names = FALSE)
  utils::write.csv(best, file.path(out_dir, 'mlr_selected_config.csv'), row.names = FALSE)

  saveRDS(list(model = fitted, recipe = recipe_fit,
               model_feature_columns = colnames(x),
               class_levels = mlr_class_levels,
               selected_config = best,
               note = 'Trained on full training partition only; no held-out test evaluation'),
          file.path(out_dir, 'mlr_selected_train_model.rds'))
  message('Selected train-only MLR exported. No test data evaluated.')
  invisible(list(selected = best, coefficients = coef_table, comparison = model_summary))
}
`%notin%` <- function(x, table) !(x %in% table)
