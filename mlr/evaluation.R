# Explicit macro-average scoring to avoid ambiguity from metric defaults.
score_mlr_predictions <- function(predictions) {
  
  levels_nqs <- c(
    "Below NQS",
    "Meeting NQS",
    "Above NQS"
  )
  
  truth <- factor(
    predictions$rating_tier,
    levels = levels_nqs
  )
  
  predicted <- factor(
    predictions$.pred_class,
    levels = levels_nqs
  )
  
  cm <- table(truth, predicted)
  
  tp <- diag(cm)
  actual_count <- rowSums(cm)
  predicted_count <- colSums(cm)
  
  precision <- ifelse(
    predicted_count > 0,
    tp / pmax(predicted_count, 1),
    0
  )
  
  recall <- ifelse(
    actual_count > 0,
    tp / pmax(actual_count, 1),
    0
  )
  
  f1 <- ifelse(
    precision + recall > 0,
    2 * precision * recall /
      pmax(precision + recall, .Machine$double.eps),
    0
  )
  
  data.frame(
    metric = c("macro_f1", "accuracy"),
    estimate = c(
      mean(f1),
      sum(tp) / sum(cm)
    )
  )
}

