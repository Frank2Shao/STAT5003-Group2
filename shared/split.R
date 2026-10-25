# Shared provider-aware split utilities (base R only).
# Inputs: data.frame with service_id, provider_id, rating_tier.
# Returns row indices; no provider can appear in more than one partition.

make_group_split <- function(data, test_prop = 0.20, seed = 5003L,
                             candidates = 400L) {
  stopifnot(all(c("service_id", "provider_id", "rating_tier") %in% names(data)))
  stopifnot(!anyNA(data$provider_id), !anyNA(data$rating_tier))
  stopifnot(test_prop > 0, test_prop < 1)
  groups <- unique(as.character(data$provider_id))
  y <- factor(data$rating_tier)
  target <- as.numeric(table(y)) * test_prop
  total_target <- nrow(data) * test_prop
  group_index <- match(as.character(data$provider_id), groups)
  group_class <- rowsum(model.matrix(~ y - 1), group_index, reorder = TRUE)
  # rowsum output indexed by sequential group numbers, in numeric order
  best_loss <- Inf
  best_test_groups <- NULL
  set.seed(seed)
  for (iteration in seq_len(candidates)) {
    chosen <- sample.int(length(groups), size = max(1, round(length(groups) * test_prop)))
    counts <- colSums(group_class[chosen, , drop = FALSE])
    loss <- sum(((counts - target) / pmax(target, 1))^2) +
      ((sum(counts) - total_target) / total_target)^2
    if (all(counts > 0) && loss < best_loss) {
      best_loss <- loss
      best_test_groups <- groups[chosen]
    }
  }
  if (is.null(best_test_groups)) stop("Unable to create split with all classes in test set")
  test_idx <- which(data$provider_id %in% best_test_groups)
  train_idx <- which(!(data$provider_id %in% best_test_groups))
  stopifnot(length(intersect(data$provider_id[train_idx],
                             data$provider_id[test_idx])) == 0)
  list(train = train_idx, test = test_idx, seed = seed,
       test_fraction = length(test_idx) / nrow(data))
}

make_group_folds <- function(train_data, v = 5L, seed = 5003L) {
  stopifnot(v >= 2, "provider_id" %in% names(train_data))
  groups <- unique(as.character(train_data$provider_id))
  if (length(groups) < v) stop("Not enough providers for requested folds")
  # Assign larger providers first; tie order is seeded. Greedily balance
  # service counts, while keeping entire providers inside a single fold.
  sizes <- as.integer(table(factor(train_data$provider_id, levels = groups)))
  set.seed(seed)
  order_groups <- order(-sizes, runif(length(groups)))
  assignment <- integer(length(groups))
  fold_size <- numeric(v)
  for (g in order_groups) {
    f <- which.min(fold_size)
    assignment[g] <- f
    fold_size[f] <- fold_size[f] + sizes[g]
  }
  fold_of_row <- assignment[match(train_data$provider_id, groups)]
  folds <- lapply(seq_len(v), function(f) {
    assessment <- which(fold_of_row == f)
    analysis <- which(fold_of_row != f)
    stopifnot(length(intersect(train_data$provider_id[analysis],
                               train_data$provider_id[assessment])) == 0)
    rsample::make_splits(list(analysis = analysis, assessment = assessment),
                         train_data)
  })
  rsample::manual_rset(folds, ids = paste0("Fold", seq_len(v)))
}

save_split_manifest <- function(data, split, output_path) {
  manifest <- data.frame(
    service_id = data$service_id, provider_id = data$provider_id,
    rating_tier = data$rating_tier,
    partition = ifelse(seq_len(nrow(data)) %in% split$test, "test", "train")
  )
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(manifest, output_path, row.names = FALSE)
  invisible(manifest)
}
