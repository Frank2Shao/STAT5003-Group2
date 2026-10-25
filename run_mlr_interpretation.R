# Execute from repository root AFTER run_mlr_tuning.R.
# RStudio: source('run_mlr_interpretation.R')
# CLI: Rscript run_mlr_interpretation.R [data.csv] [outputs/mlr]
for (pkg in c('readr', 'recipes', 'glmnet')) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop('Install missing package: ', pkg)
}
source('mlr/preprocessing.R')
source('mlr/tuning.R')
source('mlr/interpretation.R')
args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1L) args[[1L]] else 'data/processed/acecqa_analysis.csv'
out_dir <- if (length(args) >= 2L) args[[2L]] else 'outputs/mlr'
df <- as.data.frame(readr::read_csv(data_path, show_col_types = FALSE))
mlr_fit_selected_train(df, manifest_dir = out_dir, out_dir = out_dir)
