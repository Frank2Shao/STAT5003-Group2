# From project root, after baseline manifest files exist:
# source("run_mlr_tuning.R")
# Or: Rscript run_mlr_tuning.R data/processed/acecqa_analysis.csv outputs/mlr
for (pkg in c("readr", "recipes", "glmnet")) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("Install missing package: ", pkg)
}
source("mlr/preprocessing.R")
source("mlr/tuning.R")
args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1L) args[[1]] else "data/processed/acecqa_analysis.csv"
out_dir <- if (length(args) >= 2L) args[[2]] else "outputs/mlr"
df <- as.data.frame(readr::read_csv(data_path, show_col_types = FALSE))
mlr_run_tuning(df, manifest_dir = out_dir, out_dir = out_dir)
