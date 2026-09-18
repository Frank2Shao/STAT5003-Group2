required_packages <- c("readr", "dplyr", "tidyr")
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

snapshot_date <- as.Date("2026-09-18")

services_raw <- read_csv(
  "Education-services-au-export.csv",
  na = c("", "NA"),
  col_types = cols(.default = col_character()),
  name_repair = "minimal",
  show_col_types = FALSE
)

providers_raw <- read_csv(
  "Approved-providers-au-export.csv",
  na = c("", "NA"),
  col_types = cols(.default = col_character()),
  name_repair = "minimal",
  show_col_types = FALSE
)

parse_dmy_date <- function(x) {
  as.Date(as.character(x), format = "%d/%m/%Y")
}

time_to_minutes <- function(x) {
  value <- trimws(as.character(x))
  value <- gsub("\\.", ":", value)
  valid <- !is.na(value) & grepl("^[0-9]{1,2}:[0-9]{2}(:[0-9]{2})?$", value)
  result <- rep(NA_real_, length(value))
  valid_parts <- strsplit(value[valid], ":", fixed = TRUE)
  result[valid] <- vapply(valid_parts, function(parts) {
    as.numeric(parts[1]) * 60 + as.numeric(parts[2])
  }, numeric(1))
  result
}

union_daily_hours <- function(data, start_columns, end_columns) {
  start_matrix <- do.call(
    cbind,
    lapply(start_columns, function(column) time_to_minutes(data[[column]]))
  )
  end_matrix <- do.call(
    cbind,
    lapply(end_columns, function(column) time_to_minutes(data[[column]]))
  )

  if (is.null(dim(start_matrix))) {
    start_matrix <- matrix(start_matrix, ncol = 1)
    end_matrix <- matrix(end_matrix, ncol = 1)
  }

  overnight <- !is.na(start_matrix) & !is.na(end_matrix) &
    end_matrix < start_matrix
  end_matrix[overnight] <- end_matrix[overnight] + 24 * 60

  vapply(seq_len(nrow(data)), function(row_index) {
    starts <- start_matrix[row_index, ]
    ends <- end_matrix[row_index, ]
    valid <- !is.na(starts) & !is.na(ends)

    if (!any(valid)) {
      return(NA_real_)
    }

    starts <- starts[valid]
    ends <- ends[valid]
    ordering <- order(starts, ends)
    starts <- starts[ordering]
    ends <- ends[ordering]

    total_minutes <- 0
    current_start <- starts[1]
    current_end <- ends[1]

    if (length(starts) > 1) {
      for (interval_index in 2:length(starts)) {
        if (starts[interval_index] <= current_end) {
          current_end <- max(current_end, ends[interval_index])
        } else {
          total_minutes <- total_minutes + current_end - current_start
          current_start <- starts[interval_index]
          current_end <- ends[interval_index]
        }
      }
    }

    (total_minutes + current_end - current_start) / 60
  }, numeric(1))
}

normalise_time_key <- function(x) {
  x <- sub("(Start|End) Time$", "", x)
  gsub("[^A-Za-z0-9]", "", x)
}

schedule_summary <- function(data, prefix) {
  all_names <- names(data)
  start_columns <- all_names[
    grepl(paste0("^", prefix), all_names) & grepl("Start Time$", all_names)
  ]
  end_columns <- all_names[
    grepl(paste0("^", prefix), all_names) & grepl("End Time$", all_names)
  ]

  end_lookup <- setNames(end_columns, normalise_time_key(end_columns))
  paired_end_columns <- unname(end_lookup[normalise_time_key(start_columns)])
  valid_pairs <- !is.na(paired_end_columns)
  start_columns <- start_columns[valid_pairs]
  paired_end_columns <- paired_end_columns[valid_pairs]

  weekdays <- c(
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
  )

  daily_hours <- lapply(weekdays, function(day) {
    day_pairs <- grepl(day, start_columns)
    if (!any(day_pairs)) {
      return(rep(NA_real_, nrow(data)))
    }

    union_daily_hours(
      data,
      start_columns[day_pairs],
      paired_end_columns[day_pairs]
    )
  })

  daily_matrix <- as.data.frame(daily_hours)
  names(daily_matrix) <- weekdays
  open_days <- rowSums(!is.na(daily_matrix))
  weekly_hours <- rowSums(daily_matrix, na.rm = TRUE)
  weekly_hours[open_days == 0] <- NA_real_

  tibble(open_days = open_days, weekly_hours = weekly_hours)
}

row_max_or_na <- function(...) {
  values <- cbind(...)
  apply(values, 1, function(x) {
    if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)
  })
}

annual_schedule <- schedule_summary(services_raw, "Annual") %>%
  rename(
    annual_open_days = open_days,
    annual_weekly_hours = weekly_hours
  )

school_term_schedule <- schedule_summary(
  services_raw,
  "School Terms Only"
) %>%
  rename(
    school_term_open_days = open_days,
    school_term_weekly_hours = weekly_hours
  )

holiday_schedule <- schedule_summary(services_raw, "Holiday Care") %>%
  rename(
    holiday_open_days = open_days,
    holiday_weekly_hours = weekly_hours
  )

services_base <- services_raw %>%
  transmute(
    service_id = ServiceApprovalNumber,
    provider_id = `Provider Approval Number`,
    service_name = ServiceName,
    service_type = ServiceType,
    state = State,
    postcode = as.character(Postcode),
    number_approved_places = suppressWarnings(as.numeric(NumberOfApprovedPlaces)),
    service_approval_date = parse_dmy_date(ServiceApprovalGrantedDate),
    service_has_conditions = !is.na(`Conditions on Approval`),
    long_day_care = `Long Day Care`,
    preschool_part_school = `Preschool/Kindergarten - Part of a School`,
    preschool_standalone = `Preschool/Kindergarten - Stand alone`,
    oshc_after_school = `Outside school Hours Care - After School`,
    oshc_before_school = `Outside school Hours Care - Before School`,
    oshc_vacation_care = `Outside school Hours Care - Vacation Care`,
    other_service = Other,
    temporarily_closed = `Temporarily Closed`,
    overall_rating_original = OverallRating,
    rating_tier = case_when(
      OverallRating %in% c(
        "Significant Improvement Required",
        "Working Towards NQS"
      ) ~ "Below NQS",
      OverallRating == "Meeting NQS" ~ "Meeting NQS",
      OverallRating %in% c("Exceeding NQS", "Excellent") ~ "Above NQS",
      TRUE ~ NA_character_
    )
  ) %>%
  bind_cols(annual_schedule, school_term_schedule, holiday_schedule) %>%
  mutate(
    service_age_years = round(
      as.numeric(snapshot_date - service_approval_date) / 365.25,
      2
    ),
    weekly_open_days = row_max_or_na(
      annual_open_days,
      school_term_open_days,
      holiday_open_days
    ),
    max_weekly_hours = row_max_or_na(
      annual_weekly_hours,
      school_term_weekly_hours,
      holiday_weekly_hours
    ),
    schedule_count = rowSums(
      cbind(
        !is.na(annual_weekly_hours),
        !is.na(school_term_weekly_hours),
        !is.na(holiday_weekly_hours)
      )
    ),
    operating_pattern = case_when(
      schedule_count == 0 ~ "No schedule recorded",
      schedule_count > 1 ~ "Multiple schedules",
      !is.na(annual_weekly_hours) ~ "Annual",
      !is.na(school_term_weekly_hours) ~ "School terms",
      !is.na(holiday_weekly_hours) ~ "Holiday care",
      TRUE ~ NA_character_
    ),
    rating_tier = factor(
      rating_tier,
      levels = c("Below NQS", "Meeting NQS", "Above NQS")
    )
  ) %>%
  select(-schedule_count)

provider_features <- providers_raw %>%
  transmute(
    provider_id = `Provider Approval Number`,
    provider_legal_name = `Legal Name`,
    provider_trading_name = `Trading Name`,
    provider_state = State,
    provider_postcode = as.character(Postcode),
    provider_approval_date = parse_dmy_date(`Date Approval Granted`),
    provider_has_trading_name = !is.na(`Trading Name`),
    provider_has_conditions = !is.na(Conditions)
  )

provider_scale <- services_base %>%
  count(provider_id, name = "provider_service_count")

unmatched_services <- services_base %>%
  anti_join(provider_features, by = "provider_id")

integrated_data <- services_base %>%
  left_join(provider_features, by = "provider_id") %>%
  left_join(provider_scale, by = "provider_id") %>%
  mutate(
    provider_age_years = round(
      as.numeric(snapshot_date - provider_approval_date) / 365.25,
      2
    ),
    provider_state_matches_service = case_when(
      is.na(state) | is.na(provider_state) ~ NA,
      state == provider_state ~ "Yes",
      TRUE ~ "No"
    )
  )

analysis_data <- integrated_data %>%
  filter(!is.na(rating_tier)) %>%
  select(
    service_id,
    provider_id,
    rating_tier,
    service_type,
    state,
    number_approved_places,
    service_age_years,
    long_day_care,
    preschool_part_school,
    preschool_standalone,
    oshc_after_school,
    oshc_before_school,
    oshc_vacation_care,
    other_service,
    service_has_conditions,
    temporarily_closed,
    annual_open_days,
    annual_weekly_hours,
    school_term_open_days,
    school_term_weekly_hours,
    holiday_open_days,
    holiday_weekly_hours,
    weekly_open_days,
    max_weekly_hours,
    operating_pattern,
    provider_state,
    provider_age_years,
    provider_service_count,
    provider_has_trading_name,
    provider_has_conditions,
    provider_state_matches_service
  )

stopifnot(n_distinct(services_base$service_id) == nrow(services_base))
stopifnot(n_distinct(provider_features$provider_id) == nrow(provider_features))

linkage_summary <- tibble(
  measure = c(
    "Service rows",
    "Unique service IDs",
    "Provider rows",
    "Unique provider IDs in provider register",
    "Unique provider IDs represented by services",
    "Service rows unmatched to provider register",
    "Labelled analysis rows"
  ),
  value = c(
    nrow(services_raw),
    n_distinct(services_base$service_id),
    nrow(providers_raw),
    n_distinct(provider_features$provider_id),
    n_distinct(services_base$provider_id),
    nrow(unmatched_services),
    nrow(analysis_data)
  )
)

variable_dictionary <- tibble::tribble(
  ~variable, ~role, ~type, ~description,
  "service_id", "Identifier", "Character", "Unique service approval number; excluded from modelling.",
  "provider_id", "Identifier", "Character", "Provider approval number used for linkage; excluded from modelling.",
  "rating_tier", "Outcome", "Categorical", "Below NQS, Meeting NQS, or Above NQS.",
  "service_type", "Predictor", "Categorical", "Centre-Based Care or Family Day Care.",
  "state", "Predictor", "Categorical", "State or territory of the service.",
  "number_approved_places", "Predictor", "Numeric", "Maximum approved service places.",
  "service_age_years", "Predictor", "Numeric", "Years between service approval and the snapshot date.",
  "long_day_care", "Predictor", "Categorical", "Whether long day care is offered.",
  "preschool_part_school", "Predictor", "Categorical", "Whether school-based preschool or kindergarten is offered.",
  "preschool_standalone", "Predictor", "Categorical", "Whether standalone preschool or kindergarten is offered.",
  "oshc_after_school", "Predictor", "Categorical", "Whether after-school care is offered.",
  "oshc_before_school", "Predictor", "Categorical", "Whether before-school care is offered.",
  "oshc_vacation_care", "Predictor", "Categorical", "Whether vacation care is offered.",
  "other_service", "Predictor", "Categorical", "Whether another service type is recorded.",
  "service_has_conditions", "Predictor", "Logical", "Whether conditions on service approval are recorded.",
  "temporarily_closed", "Predictor", "Categorical", "Whether the service is temporarily closed.",
  "annual_open_days", "Predictor", "Numeric", "Number of days with annual operating hours.",
  "annual_weekly_hours", "Predictor", "Numeric", "Total recorded annual operating hours in a standard week.",
  "school_term_open_days", "Predictor", "Numeric", "Number of operating days during school terms.",
  "school_term_weekly_hours", "Predictor", "Numeric", "Total recorded operating hours during a school-term week.",
  "holiday_open_days", "Predictor", "Numeric", "Number of operating days during holiday care.",
  "holiday_weekly_hours", "Predictor", "Numeric", "Total recorded operating hours during a holiday-care week.",
  "weekly_open_days", "Predictor", "Numeric", "Maximum recorded open days across schedule types.",
  "max_weekly_hours", "Predictor", "Numeric", "Maximum recorded weekly hours across schedule types.",
  "operating_pattern", "Predictor", "Categorical", "Annual, school-term, holiday, multiple, or unrecorded schedule.",
  "provider_state", "Predictor", "Categorical", "State or territory recorded for the provider.",
  "provider_age_years", "Predictor", "Numeric", "Years between provider approval and the snapshot date.",
  "provider_service_count", "Predictor", "Numeric", "Number of services in the register linked to the provider.",
  "provider_has_trading_name", "Predictor", "Logical", "Whether a trading name is recorded.",
  "provider_has_conditions", "Predictor", "Logical", "Whether conditions on provider approval are recorded.",
  "provider_state_matches_service", "Predictor", "Categorical", "Whether provider and service states are the same."
)

dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
write_csv(analysis_data, "data/processed/acecqa_analysis.csv", na = "")
write_csv(linkage_summary, "data/processed/acecqa_linkage_summary.csv")
write_csv(
  variable_dictionary,
  "data/processed/acecqa_variable_dictionary.csv"
)
