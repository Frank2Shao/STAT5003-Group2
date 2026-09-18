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
  source("R/01_prepare_data.R")
}

source_dimensions <- tibble(
  dataset = c(
    "Education services register",
    "Approved providers register",
    "Integrated labelled analysis data"
  ),
  rows = c(nrow(services_raw), nrow(providers_raw), nrow(analysis_data)),
  columns = c(ncol(services_raw), ncol(providers_raw), ncol(analysis_data))
)

rating_distribution <- analysis_data %>%
  count(rating_tier, name = "count") %>%
  mutate(percent = round(100 * count / sum(count), 1))

missingness_selected <- analysis_data %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(
    everything(),
    names_to = "variable",
    values_to = "missing_count"
  ) %>%
  mutate(
    missing_percent = round(100 * missing_count / nrow(analysis_data), 2)
  ) %>%
  arrange(desc(missing_percent), variable)

raw_quality_summary <- tibble(
  measure = c(
    "Missing OverallRating values",
    "Missing NumberOfApprovedPlaces values",
    "Missing provider trading names",
    "Missing provider conditions",
    "Duplicate service IDs",
    "Duplicate provider IDs",
    "Maximum services linked to one provider"
  ),
  value = c(
    sum(is.na(services_raw$OverallRating)),
    sum(is.na(services_raw$NumberOfApprovedPlaces)),
    sum(is.na(providers_raw$`Trading Name`)),
    sum(is.na(providers_raw$Conditions)),
    sum(duplicated(services_raw$ServiceApprovalNumber)),
    sum(duplicated(providers_raw$`Provider Approval Number`)),
    max(provider_scale$provider_service_count)
  )
)

criteria_evidence <- tibble::tribble(
  ~criterion, ~evidence,
  "Large", paste0(format(nrow(services_raw), big.mark = ","), " service observations (>10,000)."),
  "Messy", paste0(
    sum(is.na(services_raw$OverallRating)),
    " missing outcomes, ",
    sum(is.na(services_raw$NumberOfApprovedPlaces)),
    " missing approved-place values, and structurally missing schedule fields."
  ),
  "Integrated", paste0(
    "Services and providers linked by Provider Approval Number; ",
    nrow(unmatched_services),
    " unmatched service rows."
  ),
  "Multi-class", "Five recorded NQS levels, consolidated into three ordered modelling tiers."
)

rating_tier_plot <- ggplot(
  rating_distribution,
  aes(
    x = factor(
      rating_tier,
      levels = c("Below NQS", "Meeting NQS", "Above NQS")
    ),
    y = count,
    fill = rating_tier
  )
) +
  geom_col(width = 0.68, show.legend = FALSE) +
  geom_text(
    aes(label = paste0(format(count, big.mark = ","), " (", percent, "%)")),
    vjust = -0.35,
    size = 3.4
  ) +
  scale_fill_manual(values = c(
    "Below NQS" = "#C44E52",
    "Meeting NQS" = "#4C78A8",
    "Above NQS" = "#59A14F"
  )) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.12)),
    labels = function(x) format(x, big.mark = ",", scientific = FALSE)
  ) +
  labs(
    title = "Recorded NQS rating tiers are imbalanced",
    x = NULL,
    y = "Number of services"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major.x = element_blank())

missingness_plot_data <- missingness_selected %>%
  filter(missing_percent > 0) %>%
  slice_max(missing_percent, n = 15, with_ties = FALSE) %>%
  arrange(missing_percent)

missingness_plot <- ggplot(
  missingness_plot_data,
  aes(x = missing_percent, y = reorder(variable, missing_percent))
) +
  geom_col(fill = "#4C78A8", width = 0.72) +
  geom_text(
    aes(label = paste0(missing_percent, "%")),
    hjust = -0.15,
    size = 3.1
  ) +
  scale_x_continuous(
    limits = c(0, max(missingness_plot_data$missing_percent) * 1.13),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "Missingness in the labelled analysis data",
    x = "Missing observations (%)",
    y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.major.y = element_blank())

dir.create("outputs", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

write_csv(source_dimensions, "outputs/source_dimensions.csv")
write_csv(rating_distribution, "outputs/rating_distribution.csv")
write_csv(missingness_selected, "outputs/missingness_selected.csv")
write_csv(raw_quality_summary, "outputs/raw_quality_summary.csv")
write_csv(criteria_evidence, "outputs/criteria_evidence.csv")

ggsave(
  "figures/rating_tier_distribution.png",
  rating_tier_plot,
  width = 7.2,
  height = 4.2,
  dpi = 220
)
ggsave(
  "figures/missingness_selected.png",
  missingness_plot,
  width = 7.2,
  height = 5.1,
  dpi = 220
)
