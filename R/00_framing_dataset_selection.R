# Framing and dataset-selection checks
# Run from the repository root:
# source("R/00_framing_dataset_selection.R")

required_packages <- c("readr", "dplyr")

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


# 1. Import the two source registers --------------------------------------

services_check <- read_csv(
  "Education-services-au-export.csv",
  na = c("", "NA"),
  col_types = cols(.default = col_character()),
  name_repair = "minimal",
  show_col_types = FALSE
)

providers_check <- read_csv(
  "Approved-providers-au-export.csv",
  na = c("", "NA"),
  col_types = cols(.default = col_character()),
  name_repair = "minimal",
  show_col_types = FALSE
)

is_non_blank <- function(x) {
  !is.na(x) & nzchar(trimws(as.character(x)))
}


# 2. Source dimensions and observation unit ------------------------------

source_dimensions_check <- tibble(
  dataset = c(
    "Education services register",
    "Approved providers register"
  ),
  rows = c(
    nrow(services_check),
    nrow(providers_check)
  ),
  columns = c(
    ncol(services_check),
    ncol(providers_check)
  )
)

service_id_check <- tibble(
  measure = c(
    "Service rows",
    "Non-missing service IDs",
    "Unique service IDs",
    "Duplicated service IDs"
  ),
  value = c(
    nrow(services_check),
    sum(is_non_blank(services_check$ServiceApprovalNumber)),
    n_distinct(
      services_check$ServiceApprovalNumber[
        is_non_blank(services_check$ServiceApprovalNumber)
      ]
    ),
    sum(duplicated(
      services_check$ServiceApprovalNumber[
        is_non_blank(services_check$ServiceApprovalNumber)
      ]
    ))
  )
)

provider_id_check <- tibble(
  measure = c(
    "Provider rows",
    "Non-missing provider IDs",
    "Unique provider IDs",
    "Duplicated provider IDs"
  ),
  value = c(
    nrow(providers_check),
    sum(is_non_blank(
      providers_check$`Provider Approval Number`
    )),
    n_distinct(
      providers_check$`Provider Approval Number`[
        is_non_blank(
          providers_check$`Provider Approval Number`
        )
      ]
    ),
    sum(duplicated(
      providers_check$`Provider Approval Number`[
        is_non_blank(
          providers_check$`Provider Approval Number`
        )
      ]
    ))
  )
)

# This field is present in the provider export but contains no usable IDs.
provider_service_id_non_missing <- sum(
  is_non_blank(providers_check$`Service Approval Number`)
)


# 3. Outcome definition --------------------------------------------------

raw_rating_distribution <- services_check |>
  count(
    OverallRating,
    name = "count",
    .drop = FALSE
  ) |>
  mutate(
    percent = round(100 * count / sum(count), 1)
  )

rating_tier_check <- case_when(
  services_check$OverallRating %in% c(
    "Significant Improvement Required",
    "Working Towards NQS"
  ) ~ "Below NQS",

  services_check$OverallRating == "Meeting NQS" ~
    "Meeting NQS",

  services_check$OverallRating %in% c(
    "Exceeding NQS",
    "Excellent"
  ) ~ "Above NQS",

  TRUE ~ NA_character_
)

rating_tier_check <- factor(
  rating_tier_check,
  levels = c(
    "Below NQS",
    "Meeting NQS",
    "Above NQS"
  )
)

rating_tier_distribution_check <- tibble(
  rating_tier = rating_tier_check
) |>
  count(
    rating_tier,
    name = "count",
    .drop = FALSE
  ) |>
  mutate(
    percent = round(100 * count / sum(count), 1)
  )


# 4. Candidate predictor availability -----------------------------------

candidate_service_fields <- c(
  "ServiceType",
  "State",
  "NumberOfApprovedPlaces",
  "ServiceApprovalGrantedDate",
  "Long Day Care",
  "Preschool/Kindergarten - Part of a School",
  "Preschool/Kindergarten - Stand alone",
  "Outside school Hours Care - After School",
  "Outside school Hours Care - Before School",
  "Outside school Hours Care - Vacation Care",
  "Other",
  "Temporarily Closed"
)

candidate_provider_fields <- c(
  "Provider Approval Number",
  "State",
  "Date Approval Granted",
  "Trading Name",
  "Conditions"
)

missing_service_fields <- setdiff(
  candidate_service_fields,
  names(services_check)
)

missing_provider_fields <- setdiff(
  candidate_provider_fields,
  names(providers_check)
)

operating_time_fields <- names(services_check)[
  grepl(
    "Start Time|End Time",
    names(services_check),
    ignore.case = TRUE
  )
]

if (length(operating_time_fields) > 0) {
  operating_time_matrix <- as.matrix(
    services_check[, operating_time_fields]
  )

  has_operating_time <- rowSums(
    !is.na(operating_time_matrix) &
      trimws(operating_time_matrix) != ""
  ) > 0
} else {
  has_operating_time <- rep(FALSE, nrow(services_check))
}

operating_time_check <- tibble(
  measure = c(
    "Operating-time columns",
    "Services with operating-time information",
    "Services without operating-time information"
  ),
  value = c(
    length(operating_time_fields),
    sum(has_operating_time),
    sum(!has_operating_time)
  )
)


# 5. Linkage-key validation ----------------------------------------------

service_provider_ids <- services_check$`Provider Approval Number`
registered_provider_ids <-
  providers_check$`Provider Approval Number`

provider_match <- service_provider_ids %in%
  registered_provider_ids

linkage_check <- tibble(
  measure = c(
    "Service rows",
    "Matched service rows",
    "Unmatched service rows"
  ),
  value = c(
    nrow(services_check),
    sum(provider_match),
    sum(!provider_match)
  )
)


# 6. Leakage identification ---------------------------------------------

rating_or_visit_fields <- names(services_check)[
  grepl(
    "Rating|Visit",
    names(services_check),
    ignore.case = TRUE
  )
]

excluded_from_modelling <- unique(c(
  "ServiceApprovalNumber",
  "Provider Approval Number",
  rating_or_visit_fields
))

eligible_model_fields <- setdiff(
  names(services_check),
  excluded_from_modelling
)


# 7. Evidence for dataset-selection criteria ----------------------------

dataset_criteria_check <- tibble(
  criterion = c(
    "Large",
    "Integrated",
    "Multi-class",
    "Messy"
  ),
  evidence = c(
    paste(
      format(nrow(services_check), big.mark = ","),
      "service observations"
    ),
    paste(
      nrow(services_check),
      "service rows linked to",
      nrow(providers_check),
      "provider records using Provider Approval Number"
    ),
    paste(
      n_distinct(
        as.character(
          rating_tier_check[!is.na(rating_tier_check)]
        )
      ),
      "modelled outcome classes"
    ),
    paste(
      sum(is.na(services_check$OverallRating)),
      "missing outcomes and",
      sum(is.na(
        services_check$NumberOfApprovedPlaces
      )),
      "missing approved-place values"
    )
  )
)


# 8. Reproducibility assertions -----------------------------------------

expected_rating_tiers <- c(
  "Below NQS",
  "Meeting NQS",
  "Above NQS"
)

observed_rating_tiers <- unique(
  as.character(
    rating_tier_check[!is.na(rating_tier_check)]
  )
)

stopifnot(
  # One row represents one unique approved service.
  nrow(services_check) ==
    n_distinct(services_check$ServiceApprovalNumber),

  all(is_non_blank(
    services_check$ServiceApprovalNumber
  )),

  # One row in the provider register represents one provider.
  nrow(providers_check) ==
    n_distinct(
      providers_check$`Provider Approval Number`
    ),

  all(is_non_blank(
    providers_check$`Provider Approval Number`
  )),

  # The provider export's service-ID field is unusable.
  provider_service_id_non_missing == 0,

  # Every service can be linked to a provider.
  all(provider_match),

  # All required candidate fields are available.
  length(missing_service_fields) == 0,
  length(missing_provider_fields) == 0,
  length(operating_time_fields) > 0,

  # The constructed outcome has exactly three intended levels.
  setequal(
    observed_rating_tiers,
    expected_rating_tiers
  ),

  # Every non-missing original rating is mapped.
  sum(is.na(rating_tier_check)) ==
    sum(is.na(services_check$OverallRating)),

  # Leakage-prone fields are excluded from eligible predictors.
  !any(
    grepl(
      "Rating|Visit",
      eligible_model_fields,
      ignore.case = TRUE
    )
  )
)


# 9. Display audit results -----------------------------------------------

cat("\nSource dimensions\n")
print(source_dimensions_check)

cat("\nService identifier checks\n")
print(service_id_check)

cat("\nProvider identifier checks\n")
print(provider_id_check)

cat("\nRaw rating distribution\n")
print(raw_rating_distribution)

cat("\nThree-level outcome distribution\n")
print(rating_tier_distribution_check)

cat("\nProvider linkage checks\n")
print(linkage_check)

cat("\nOperating-time availability\n")
print(operating_time_check)

cat("\nDataset-selection evidence\n")
print(dataset_criteria_check)

cat("\nRating- and visit-derived fields excluded from modelling\n")
print(rating_or_visit_fields)

cat("\nAll framing and dataset-selection checks passed.\n")