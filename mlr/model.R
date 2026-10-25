# Ridge-regularised multinomial logistic regression using glmnet.
make_mlr_spec <- function(penalty = 0.01) {
  parsnip::multinom_reg(penalty = penalty, mixture = 0) |>
    parsnip::set_engine("glmnet") |>
    parsnip::set_mode("classification")
}

make_mlr_workflow <- function(training_data, penalty = 0.01) {
  workflows::workflow() |>
    workflows::add_recipe(make_mlr_recipe(training_data)) |>
    workflows::add_model(make_mlr_spec(penalty))
}
