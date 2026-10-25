# MLR-only features and leakage-safe preprocessing.
# recipe is prepped separately inside each CV analysis fold by workflows.

mlr_feature_names <- function() {
  c("service_type", "state", "number_approved_places", "service_age_years",
    "provider_age_years", "provider_service_count", "operating_pattern",
    "max_weekly_hours", "long_day_care", "preschool_part_school",
    "preschool_standalone", "oshc_after_school", "oshc_before_school",
    "oshc_vacation_care", "other_service", "service_has_conditions",
    "temporarily_closed", "provider_has_conditions",
    "provider_has_trading_name")
}

mlr_model_data <- function(data) {
  cols <- c("rating_tier", mlr_feature_names())
  missing <- setdiff(cols, names(data))
  if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))
  out <- data[, cols, drop = FALSE]
  out$rating_tier <- factor(out$rating_tier,
                            levels = c("Below NQS", "Meeting NQS", "Above NQS"))
  if (anyNA(out$rating_tier)) stop("Unexpected outcome labels")
  # Cast logical Yes/No flags to factors; retain numeric measures as numeric.
  for (name in names(out)) {
    if (name != "rating_tier" && (is.logical(out[[name]]) || is.character(out[[name]]))) {
      out[[name]] <- factor(out[[name]])
    }
  }
  out
}

make_mlr_recipe <- function(training_data) {
  recipes::recipe(rating_tier ~ ., data = training_data) |>
    recipes::step_impute_median(recipes::all_numeric_predictors()) |>
    recipes::step_unknown(recipes::all_nominal_predictors(), new_level = "missing") |>
    recipes::step_novel(recipes::all_nominal_predictors()) |>
    recipes::step_other(recipes::all_nominal_predictors(), threshold = 0.01) |>
    recipes::step_dummy(recipes::all_nominal_predictors()) |>
    recipes::step_zv(recipes::all_predictors()) |>
    recipes::step_normalize(recipes::all_numeric_predictors())
}
