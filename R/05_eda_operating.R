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

operating_data <- read_csv(
  "data/processed/acecqa_analysis.csv",
  show_col_types = FALSE
)

operating_variables <- c(
  "annual_open_days",
  "annual_weekly_hours",
  "school_term_open_days",
  "school_term_weekly_hours",
  "holiday_open_days",
  "holiday_weekly_hours",
  "weekly_open_days",
  "max_weekly_hours",
  "operating_pattern"
)

operating_variables %in% names(operating_data)

dim(operating_data)

# Numerical operating variables
operating_numeric <- c(
  "annual_open_days",
  "annual_weekly_hours",
  "school_term_open_days",
  "school_term_weekly_hours",
  "holiday_open_days",
  "holiday_weekly_hours",
  "weekly_open_days",
  "max_weekly_hours"
)

# Create a basic summary table
operating_summary <- data.frame(
  variable = operating_numeric,

  observed = sapply(
    operating_data[operating_numeric],
    function(x) sum(!is.na(x))
  ),

  missing = sapply(
    operating_data[operating_numeric],
    function(x) sum(is.na(x))
  ),

  missing_percent = sapply(
    operating_data[operating_numeric],
    function(x) round(mean(is.na(x)) * 100, 1)
  ),

  minimum = sapply(
    operating_data[operating_numeric],
    function(x) min(x, na.rm = TRUE)
  ),

  median = sapply(
    operating_data[operating_numeric],
    function(x) median(x, na.rm = TRUE)
  ),

  maximum = sapply(
    operating_data[operating_numeric],
    function(x) max(x, na.rm = TRUE)
  )
)

#print(operating_summary)

# Check questionable or extreme operating-hour values
operating_checks <- data.frame(
  check = c(
    "Missing max weekly hours",
    "Zero max weekly hours",
    "More than 100 hours per week",
    "More than 168 hours per week",
    "Open days outside 0 to 7"
  ),

  count = c(
    sum(is.na(operating_data$max_weekly_hours)),
    sum(operating_data$max_weekly_hours == 0, na.rm = TRUE),
    sum(operating_data$max_weekly_hours > 100, na.rm = TRUE),
    sum(operating_data$max_weekly_hours > 168, na.rm = TRUE),
    sum(
      operating_data$weekly_open_days < 0 |
      operating_data$weekly_open_days > 7,
      na.rm = TRUE
    )
  )
)

#print(operating_checks)

# Display the ten services with the longest weekly hours
longest_hours <- operating_data |>
  select(
    service_id,
    service_type,
    rating_tier,
    weekly_open_days,
    max_weekly_hours,
    operating_pattern
  ) |>
  arrange(desc(max_weekly_hours)) |>
  head(10)

#print(longest_hours)

# Overall summaries by rating tier
operating_by_tier <- operating_data |>
  group_by(rating_tier) |>
  summarise(
    services = n(),
    missing_hours = sum(is.na(max_weekly_hours)),
    median_open_days = median(weekly_open_days, na.rm = TRUE),
    median_weekly_hours = median(max_weekly_hours, na.rm = TRUE),
    q1_weekly_hours = quantile(
      max_weekly_hours,
      0.25,
      na.rm = TRUE
    ),
    q3_weekly_hours = quantile(
      max_weekly_hours,
      0.75,
      na.rm = TRUE
    )
  )

#print(operating_by_tier)

# Sensitivity check within each service type
operating_by_type_tier <- operating_data |>
  group_by(service_type, rating_tier) |>
  summarise(
    services = n(),
    missing_hours = sum(is.na(max_weekly_hours)),
    median_open_days = median(weekly_open_days, na.rm = TRUE),
    median_weekly_hours = median(max_weekly_hours, na.rm = TRUE),
    .groups = "drop"
  )

#print(operating_by_type_tier)

# Set the logical order of the outcome
operating_data$rating_tier <- factor(
  operating_data$rating_tier,
  levels = c("Below NQS", "Meeting NQS", "Above NQS")
)

# Distribution of weekly open days
open_days_distribution <- operating_data |>
  count(weekly_open_days) |>
  mutate(
    percent = round(100 * n / sum(n), 1)
  )

print(open_days_distribution)

# Relationship between open days and weekly hours
overall_operating_correlation <- cor(
  operating_data$weekly_open_days,
  operating_data$max_weekly_hours,
  use = "complete.obs",
  method = "spearman"
)

print(overall_operating_correlation)

# Correlation within each service type
correlation_by_service_type <- operating_data |>
  group_by(service_type) |>
  summarise(
    services = n(),
    spearman_correlation = cor(
      weekly_open_days,
      max_weekly_hours,
      use = "complete.obs",
      method = "spearman"
    )
  )

print(correlation_by_service_type)

# Colours consistent with the other EDA sections
rating_colours <- c(
  "Below NQS" = "#C44E52",
  "Meeting NQS" = "#4C78A8",
  "Above NQS" = "#59A14F"
)

# Weekly operating hours by rating tier and service type
operating_hours_plot <- ggplot(
  operating_data |>
    filter(!is.na(max_weekly_hours)),
  aes(
    x = rating_tier,
    y = max_weekly_hours,
    fill = rating_tier
  )
) +
  geom_boxplot(
    width = 0.6,
    outlier.alpha = 0.2,
    outlier.size = 0.8
  ) +
  facet_wrap(~ service_type) +
  scale_fill_manual(values = rating_colours) +
  scale_y_sqrt() +
  labs(
    title = "Weekly operating hours by recorded NQS tier",
    subtitle = "Square-root scale retains the full range while compressing extreme values",
    x = NULL,
    y = "Maximum recorded weekly hours",
    fill = "Rating tier"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold")
  )

print(operating_hours_plot)

dir.create("figures", showWarnings = FALSE)

ggsave(
  "figures/operating_hours_by_rating.png",
  operating_hours_plot,
  width = 9,
  height = 5,
  dpi = 180
)

# Save supporting outputs
dir.create(
  "outputs/operating",
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  operating_summary,
  "outputs/operating/operating_summary.csv"
)

write_csv(
  operating_checks,
  "outputs/operating/operating_checks.csv"
)

write_csv(
  operating_by_tier,
  "outputs/operating/operating_by_tier.csv"
)

write_csv(
  operating_by_type_tier,
  "outputs/operating/operating_by_type_tier.csv"
)

write_csv(
  open_days_distribution,
  "outputs/operating/open_days_distribution.csv"
)

write_csv(
  correlation_by_service_type,
  "outputs/operating/correlation_by_service_type.csv"
)