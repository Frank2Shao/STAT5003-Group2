# Run once from the repository root. Keep dependencies inside this project.
if (!file.exists("R/06_model_setup.R")) stop("Run from the group repository root.")
dir.create(".R-library", showWarnings = FALSE)
.libPaths(c(normalizePath(".R-library"), .libPaths()))
packages <- c("readr", "dplyr", "caret", "recipes", "kknn", "ggplot2", "rmarkdown", "knitr")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org",
                                    type = if (.Platform$OS.type == "windows") "binary" else getOption("pkgType"))
