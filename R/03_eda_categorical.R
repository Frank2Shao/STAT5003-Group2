required_packages <- c("readr", "dplyr", "tidyr", "ggplot2")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Install the following packages before running this script: ",
    paste(missing_packages, collapse = ", ")
  )
}

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

if (!exists("analysis_data")) {
  source("R/02_data_quality.R")
}

# ---Select Categorical Variable---
categorical_data <- analysis_data |>
  select(all_of(
    variable_dictionary |> 
      filter(role == "Outcome"|(role == "Predictor" & type %in% c("Categorical", "Logical"))) |>
      pull(variable)
  ))
categorical_vars <- setdiff(names(categorical_data), "rating_tier")

categorical_data_clean <- categorical_data |>
  mutate(across(
    all_of(categorical_vars),
    ~ ifelse(is.na(.x), "Missing", as.character(.x))
  )) 
categorical_data_long <- categorical_data_clean |>
  pivot_longer(
    all_of(categorical_vars),
    names_to  = "variable",
    values_to = "level"
  )

# ---Variable Profile---
categorical_profile <- categorical_data_long |>
  count(variable, level, name = "count") |>
  group_by(variable) |>
  summarise(
    n_levels = length(level),
    missing_count = sum(count[level=="Missing"]),
    top_level = level[which.max(count)],
    top_percent = round(100*max(count)/sum(count), 1),
    smallest_level = level[which.min(count)],
    smallest_count = min(count),
    rare_levels = sum(count/sum(count) < 0.01),
    .groups = "drop"
  ) |>
  mutate(
    missing_percent = round(100*missing_count/nrow(categorical_data_clean), 2),
    flag = case_when(
      n_levels == 1 ~ "Constant - drop",
      top_percent >= 95 ~ "Near-constant (>=95% one level)",
      rare_levels > 0 ~ "Has rare levels (<1%) - consider collapsing",
      missing_percent > 20 ~ "High missingness",
      TRUE ~ "OK"
    )
  ) |>
  arrange(desc(top_percent))

# ---Univariate Frequencies---
categorical_freq_table <- categorical_data_long |>
  count(variable, level, name = "count") |>
  group_by(variable) |>
  mutate(percent = round(100*count/sum(count))) |>
  arrange(variable, desc(count), .by_group = TRUE) |>
  ungroup()

categorical_univariate_plot <- ggplot(
  categorical_freq_table, aes(x=percent, y=level)
) + 
  geom_col(fill = "#4C78A8", width = 0.7) +
  geom_text(aes(label = paste0(percent, "%")), hjust = -0.15, size = 2.8) +
  facet_wrap(~variable, scales = "free_y", ncol = 3)+
  scale_x_continuous(
    limits = c(0,115),
    breaks = c(0,50,100),
    expand = expansion(mult = c(0,0))
  ) +
  labs(
    title = "Level frequencies of categorical variables",
    x = "Share of services (%)",
    y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.major.y = element_blank())

# ---Predictors vs. Outcome---
# Row Percentages
categorical_outcome_crosstab <- categorical_data_long |>
  count(variable, level, rating_tier, name = "count") |>
  group_by(variable, level) |>
  mutate(
    level_total = sum(count),
    row_percent = round(100*count/level_total, 1)
  ) |>
  ungroup()
categorical_overall_rating <- categorical_data_clean |>
  count(rating_tier, name = "count") |>
  mutate(overall_percent = round(100*count/sum(count), 1))

# Cramer's V to check for effect size
categorical_cramers_v <- function(x, y){
  tab <- table(x, y)
  tab <- tab[rowSums(tab) > 0, colSums(tab) > 0, drop = FALSE]
  r <- nrow(tab)
  k <- ncol(tab)
  n <- sum(tab)
  if (r < 2 || k < 2) {return(NA_real_)}
  chi2 <- as.numeric(suppressWarnings(chisq.test(tab, correct = FALSE)$statistic))
  phi2_corr <- max(0, chi2/n - (r-1)*(k-1)/(n-1))
  r_corr <- r - (r-1)^2/(n-1)
  k_corr <- k - (k-1)^2/(n-1)
  sqrt(phi2_corr/min(r_corr-1, k_corr-1))
}
categorical_association_with_outcome <- lapply(categorical_vars, function(v){
  tab <- table(categorical_data_clean[[v]], categorical_data_clean[["rating_tier"]])
  if (nrow(tab) < 2) {return(tibble(variable = v, n_levels = nrow(tab)))}
  test <- suppressWarnings(chisq.test(tab))
  tibble(
    variable = v,
    n_levels = nrow(tab),
    chi_square = unname(test$statistic),
    df = unname(test$parameter),
    p.value = test$p.value,
    cramers_v = categorical_cramers_v(
      categorical_data_clean[[v]],
      categorical_data_clean[["rating_tier"]]
    ),
    pct_cells_expected_lt5 = round(100*mean(test$expected < 5), 1)
  )
}) |>
  bind_rows() |>
  mutate(
    p_adjusted = p.adjust(p.value, method = "holm"),
    strength = cut(
      cramers_v,
      breaks = c(-Inf, 0.1, 0.2, 0.4, Inf),
      labels = c("Neglibile", "Weak", "Moderate", "Strong")
    ),
    reliability_note = if_else(
      pct_cells_expected_lt5 > 20,
      "Sparse cells: collapse rare levels before trusting chi-square",
      ""
    )
  ) |>
  arrange(desc(cramers_v))

# Standardized Pearson Residuals
categorical_std_residuals <- lapply(categorical_vars, function(v){
  tab <- table(
    level = categorical_data_clean[[v]],
    rating_tier = categorical_data_clean[["rating_tier"]]
  )
  if (nrow(tab) < 2) {return(NULL)}
  residuals <- suppressWarnings(chisq.test(tab)$stdres)
  as_tibble(as.data.frame(as.table(residuals), stringAsFactors = FALSE)) |>
    rename(std_residual = Freq) |>
    mutate(variable = v, std_residual = round(std_residual, 2))
}) |>
  bind_rows() |>
  select(variable, level, rating_tier, std_residual)

# Stacked bars for each predictor, ordered by Cramer's V
categorical_facet_labels <- categorical_association_with_outcome |>
  transmute(variable, facet_label = paste0(variable, " (V = ", round(cramers_v, 2), ")"))
categorical_crosstab_plot_data <- categorical_outcome_crosstab |>
  left_join(categorical_facet_labels, by = "variable") |>
  mutate(
    facet_label = factor(facet_label, levels = categorical_facet_labels$facet_label),
    rating_tier = factor(
      rating_tier,
      levels = c("Below NQS", "Meeting NQS", "Above NQS")
    )
  )
categorical_outcome_plot <- ggplot(
  categorical_crosstab_plot_data,
  aes(x = count, y = level, fill = rating_tier)
) +
  geom_col(position = "fill", width = 0.75) +
  facet_wrap(~facet_label, scales = "free_y", ncol = 3) +
  scale_fill_manual(
    values = c(
      "Below NQS" = "#C44E52",
      "Meeting NQS" = "#4C78A8",
      "Above NQS" = "#59A14F"
    )
  ) +
  scale_x_continuous(labels = scales::percent_format()) +
  labs(
    title = "Rating Tier Mix within Each Level of Each Categorical Predictor",
    subtitle = "Panels ordered by strength of association (Cramer's V)",
    x = "Share of Services",
    y = NULL,
    fill = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.major.y = element_blank(),
    legend.position = "top"
  )
categorical_top_residual_vars <- categorical_association_with_outcome |>
  filter(!is.na(cramers_v)) |>
  slice_head(n = 7) |>
  pull(variable)
categorical_residual_plot <- categorical_std_residuals |>
  filter(variable %in% categorical_top_residual_vars) |>
  mutate(
    rating_tier = factor(
      rating_tier,
      levels = c("Below NQS", "Meeting NQS", "Above NQS")
    )
  ) |>
  ggplot(aes(x = rating_tier, y = level, fill = std_residual)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = round(std_residual, 1)), size = 2.8) +
  facet_wrap(~variable, scales = "free_y", ncol = 2) +
  scale_fill_gradient2(
    low = "#4C78A8",
    mid = "white",
    high = "#C44E52",
    midpoint = 0,
    limits = c(-10,10),
    oob = scales::squish
  ) +
  labs(
    title = "Where the Association Comes From",
    subtitle = "Standardised residuals: Red = More services than expected, Blue = Fewer",
    x = NULL,
    y = NULL,
    fill = "Std. Residual"
  ) +
  theme_minimal(base_size = 10) +
  theme(panel.grid = element_blank())

# ---Predictor vs. Predictor - Redundancy Check---
categorical_pairwise_association <- expand.grid(
  var1 = categorical_vars,
  var2 = categorical_vars,
  stringsAsFactors = FALSE
) |>
  as_tibble() |>
  mutate(
    cramers_v = mapply(
      function(a,b) {
        if (a==b) 1 else categorical_cramers_v(categorical_data_clean[[a]], categorical_data_clean[[b]])
      },
      var1,
      var2
    )
  )

categorical_redundant_pairs <- categorical_pairwise_association |>
  filter(var1 < var2, !is.na(cramers_v), cramers_v >= 0.5) |>
  arrange(desc(cramers_v))

categorical_pairwise_plot <- ggplot(
  categorical_pairwise_association,
  aes(x = var1, y = var2, fill=cramers_v)
) +
  geom_tile(colour = "white") +
  geom_text(aes(label = round(cramers_v, 2)), size = 2.6) +
  scale_fill_gradient(
    low = "white",
    high = "#4C78A8",
    limits = c(0, 1),
    na.value = "grey90"
  ) +
  labs(
    title = "Association Between Categorical Predictors",
    subtitle = "Bias-corrected Cramer's V; High values = Redundant features",
    x = NULL,
    y = NULL,
    fill = "V"
  ) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.grid = element_blank())

# ---Saving Outputs---
dir.create("outputs/categorical", showWarnings = FALSE)
write_csv(categorical_profile, "outputs/categorical/profile.csv")
write_csv(categorical_freq_table, "outputs/categorical/frequencies.csv")
write_csv(categorical_outcome_crosstab, "outputs/categorical/categorical_vs_rating_crosstab.csv")
write_csv(categorical_association_with_outcome, "outputs/categorical/standardised_residuals.csv")
write_csv(categorical_pairwise_association, "outputs/categorical/pairwise_cramers_v.csv")
write_csv(categorical_redundant_pairs, "outputs/categorical/redundant_pairs.csv")

ggsave("figures/categorical_univariate.png", categorical_univariate_plot,
       width = 10, height = 9, dpi = 220)
ggsave("figures/categorical_vs_rating_tier.png", categorical_outcome_plot,
       width = 11, height = 10, dpi = 220)
ggsave("figures/categorical_residuals.png", categorical_residual_plot,
       width = 9, height = 9, dpi = 220)
ggsave("figures/categorical_pairwise_cramers_v.png", categorical_pairwise_plot,
       width = 8.5, height = 7.5, dpi = 220)