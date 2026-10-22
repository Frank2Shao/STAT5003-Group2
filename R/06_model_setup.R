# Shared modelling workflow
# Creates the common outcome, predictor set, train/test split,
# resampling folds, preprocessing specification, and evaluation metrics.SS

required_packages <- c(
  "readr",
  "dplyr",
  "caret",
  "recipes"
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


model_data_raw <- read_csv(
  "data/processed/acecqa_analysis.csv",
  show_col_types = FALSE
)

required_columns <- c(
  "service_id",
  "provider_id",
  "rating_tier"
)

missing_columns <- setdiff(
  required_columns,
  names(model_data_raw)
)

if (length(missing_columns) > 0) {
  stop(
    "Required columns are missing: ",
    paste(missing_columns, collapse = ", ")
  )
}

model_data_raw <- model_data_raw |>
  mutate(
    rating_tier = factor(
      rating_tier,
      levels = c(
        "Below NQS",
        "Meeting NQS",
        "Above NQS"
      )
    )
  )

stopifnot(
  nrow(model_data_raw) > 0,
  !anyDuplicated(model_data_raw$service_id),
  !anyNA(model_data_raw$provider_id),
  !anyNA(model_data_raw$rating_tier)
)

# 80/20 train-test split

split_seed <- 5003L
training_fraction <- 0.80

# A provider may operate services in different rating tiers.
# Use its most frequent tier only as an approximate stratification label.
provider_split_frame <- model_data_raw |>
  count(provider_id, rating_tier, name = "n_services") |>
  group_by(provider_id) |>
  arrange(desc(n_services), rating_tier, .by_group = TRUE) |>
  slice(1L) |>
  ungroup() |>
  rename(provider_stratum = rating_tier)

set.seed(split_seed)

training_provider_rows <- createDataPartition(
  provider_split_frame$provider_stratum,
  p = training_fraction,
  list = FALSE
)

training_provider_ids <- provider_split_frame$provider_id[
  as.vector(training_provider_rows)
]

training_data <- model_data_raw |>
  filter(provider_id %in% training_provider_ids)

test_data <- model_data_raw |>
  filter(!(provider_id %in% training_provider_ids))

# Confirm that providers do not appear in both datasets.
stopifnot(
  length(intersect(
    unique(training_data$provider_id),
    unique(test_data$provider_id)
  )) == 0,
  nrow(training_data) + nrow(test_data) == nrow(model_data_raw)
)

# Inspect the realised split and outcome distributions.
split_distribution <- bind_rows(
  training_data |> mutate(split = "Training"),
  test_data |> mutate(split = "Test")
) |>
  count(split, rating_tier, name = "n_services") |>
  group_by(split) |>
  mutate(
    percentage = round(100 * n_services / sum(n_services), 1)
  ) |>
  ungroup()

print(split_distribution)



# Provider-grouped five-fold cross-validation



cv_seed <- 5004L
n_folds <- 5L

provider_cv_frame <- provider_split_frame |>
  filter(provider_id %in% training_provider_ids)

set.seed(cv_seed)

provider_validation_folds <- createFolds(
  provider_cv_frame$provider_stratum,
  k = n_folds,
  returnTrain = FALSE
)

# Convert provider-level folds into service-row indices.
cv_index_out <- lapply(
  provider_validation_folds,
  function(provider_rows) {
    validation_provider_ids <- provider_cv_frame$provider_id[provider_rows]

    which(training_data$provider_id %in% validation_provider_ids)
  }
)

# caret requires the complementary training indices.
all_training_rows <- seq_len(nrow(training_data))

cv_index <- lapply(
  cv_index_out,
  function(validation_rows) {
    setdiff(all_training_rows, validation_rows)
  }
)

names(cv_index) <- paste0("Fold", seq_len(n_folds))
names(cv_index_out) <- names(cv_index)

# Confirm that each service appears in exactly one validation fold
# and that no provider crosses training and validation within a fold.
stopifnot(
  all(
    tabulate(
      unlist(cv_index_out),
      nbins = nrow(training_data)
    ) == 1L
  )
)

for (fold_name in names(cv_index)) {
  analysis_providers <- unique(
    training_data$provider_id[cv_index[[fold_name]]]
  )

  validation_providers <- unique(
    training_data$provider_id[cv_index_out[[fold_name]]]
  )

  stopifnot(
    length(intersect(
      analysis_providers,
      validation_providers
    )) == 0
  )
}

cv_fold_summary <- bind_rows(
  lapply(
    names(cv_index_out),
    function(fold_name) {
      training_data[cv_index_out[[fold_name]], ] |>
        count(rating_tier, name = "n_services") |>
        mutate(fold = fold_name)
    }
  )
) |>
  group_by(fold) |>
  mutate(
    percentage = round(100 * n_services / sum(n_services), 1)
  ) |>
  ungroup() |>
  select(fold, rating_tier, n_services, percentage)

print(cv_fold_summary)

# -------------------------------------------------------------------
# Common outcome and predictor set
# -------------------------------------------------------------------

variable_dictionary <- read_csv(
  "data/processed/acecqa_variable_dictionary.csv",
  show_col_types = FALSE
)

predictor_names <- variable_dictionary |>
  filter(role == "Predictor") |>
  pull(variable)

missing_predictors <- setdiff(
  predictor_names,
  names(model_data_raw)
)

if (length(missing_predictors) > 0) {
  stop(
    "Predictor columns are missing: ",
    paste(missing_predictors, collapse = ", ")
  )
}

training_model_data <- training_data |>
  select(
    rating_tier,
    all_of(predictor_names)
  )

test_model_data <- test_data |>
  select(
    rating_tier,
    all_of(predictor_names)
  )

# Confirm that identifiers and rating-derived variables are excluded.
forbidden_predictors <- c(
  "service_id",
  "provider_id",
  "overall_rating_original"
)

stopifnot(
  !any(forbidden_predictors %in% names(training_model_data)),
  identical(
    names(training_model_data),
    names(test_model_data)
  ),
  ncol(training_model_data) == length(predictor_names) + 1L
)

cat(
  "\nOutcome: rating_tier",
  "\nNumber of predictors:", length(predictor_names),
  "\nTraining observations:", nrow(training_model_data),
  "\nTest observations:", nrow(test_model_data),
  "\n"
)

# -------------------------------------------------------------------
# Fold-wise preprocessing specifications
# -------------------------------------------------------------------

logical_predictors <- variable_dictionary |>
  filter(
    role == "Predictor",
    type == "Logical"
  ) |>
  pull(variable)

# Convert logical predictors using fixed, predefined levels.
training_model_data <- training_model_data |>
  mutate(
    across(
      all_of(logical_predictors),
      ~ factor(
        .x,
        levels = c(FALSE, TRUE),
        labels = c("No", "Yes")
      )
    )
  )

test_model_data <- test_model_data |>
  mutate(
    across(
      all_of(logical_predictors),
      ~ factor(
        .x,
        levels = c(FALSE, TRUE),
        labels = c("No", "Yes")
      )
    )
  )

# Operations shared by all five models.
base_recipe <- recipe(
  rating_tier ~ .,
  data = training_model_data
) |>
  step_string2factor(
    all_nominal_predictors()
  ) |>
  step_unknown(
    all_nominal_predictors(),
    new_level = "Unknown"
  ) |>
  step_novel(
    all_nominal_predictors(),
    new_level = "Novel"
  ) |>
  step_other(
    all_nominal_predictors(),
    threshold = 0.01,
    other = "Other"
  ) |>
  step_impute_median(
    all_numeric_predictors()
  ) |>
  step_log(
    provider_service_count,
    offset = 1
  )

# For multinomial logistic regression, LDA, kNN and SVM.
scaled_recipe <- base_recipe |>
  step_normalize(
    all_numeric_predictors()
  ) |>
  step_dummy(
    all_nominal_predictors()
  ) |>
  step_nzv(
    all_predictors()
  )

# For Random Forest, where scaling is unnecessary.
tree_recipe <- base_recipe |>
  step_dummy(
    all_nominal_predictors()
  ) |>
  step_nzv(
    all_predictors()
  )

print(scaled_recipe)
print(tree_recipe)

# -------------------------------------------------------------------
# Shared evaluation function and caret resampling control
# -------------------------------------------------------------------

macro_f1_summary <- function(data, lev = NULL, model = NULL) {
  if (is.null(lev)) {
    lev <- levels(data$obs)
  }

  f1_by_class <- vapply(
    lev,
    function(class_name) {
      true_positive <- sum(
        data$pred == class_name &
          data$obs == class_name
      )

      false_positive <- sum(
        data$pred == class_name &
          data$obs != class_name
      )

      false_negative <- sum(
        data$pred != class_name &
          data$obs == class_name
      )

      precision_denominator <- true_positive + false_positive
      recall_denominator <- true_positive + false_negative

      precision <- if (precision_denominator == 0) {
        0
      } else {
        true_positive / precision_denominator
      }

      recall <- if (recall_denominator == 0) {
        0
      } else {
        true_positive / recall_denominator
      }

      if (precision + recall == 0) {
        0
      } else {
        2 * precision * recall / (precision + recall)
      }
    },
    numeric(1)
  )

  recall_by_class <- vapply(
    lev,
    function(class_name) {
      denominator <- sum(data$obs == class_name)

      if (denominator == 0) {
        0
      } else {
        sum(
          data$pred == class_name &
            data$obs == class_name
        ) / denominator
      }
    },
    numeric(1)
  )

  names(recall_by_class) <- paste0(
    "Recall_",
    make.names(lev)
  )

  c(
    Macro_F1 = mean(f1_by_class),
    Accuracy = mean(data$pred == data$obs),
    recall_by_class
  )
}

model_control <- trainControl(
  method = "cv",
  number = n_folds,
  index = cv_index,
  indexOut = cv_index_out,
  classProbs = FALSE,
  summaryFunction = macro_f1_summary,
  savePredictions = "final",
  returnResamp = "final",
  allowParallel = TRUE
)

# Check the evaluation function using a majority-class baseline.
majority_class <- names(
  which.max(
    table(training_model_data$rating_tier)
  )
)

baseline_check <- data.frame(
  obs = training_model_data$rating_tier,
  pred = factor(
    rep(
      majority_class,
      nrow(training_model_data)
    ),
    levels = levels(training_model_data$rating_tier)
  )
)

baseline_metrics <- macro_f1_summary(
  baseline_check,
  lev = levels(training_model_data$rating_tier)
)

print(baseline_metrics)


fold1_training <- training_model_data[
  cv_index[["Fold1"]],
  ,
  drop = FALSE
]

fold1_validation <- training_model_data[
  cv_index_out[["Fold1"]],
  ,
  drop = FALSE
]

# Learn preprocessing parameters from Fold 1 training data only.
scaled_recipe_check <- prep(
  scaled_recipe,
  training = fold1_training
)

tree_recipe_check <- prep(
  tree_recipe,
  training = fold1_training
)

# Apply the learned parameters to Fold 1 validation data.
scaled_validation_check <- bake(
  scaled_recipe_check,
  new_data = fold1_validation
)

tree_validation_check <- bake(
  tree_recipe_check,
  new_data = fold1_validation
)

stopifnot(
  nrow(scaled_validation_check) == nrow(fold1_validation),
  nrow(tree_validation_check) == nrow(fold1_validation),
  !anyNA(scaled_validation_check),
  !anyNA(tree_validation_check)
)

cat(
  "Scaled recipe output:",
  nrow(scaled_validation_check), "rows ×",
  ncol(scaled_validation_check), "columns\n",
  "Tree recipe output:",
  nrow(tree_validation_check), "rows ×",
  ncol(tree_validation_check), "columns\n"
)