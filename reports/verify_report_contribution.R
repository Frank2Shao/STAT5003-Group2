# Run from the repository root after rendering both reports.
read <- function(path) read.csv(path, check.names = FALSE)
cv <- read("outputs/random_forest/cv_summary.csv")
folds <- read("outputs/random_forest/cv_fold_metrics.csv")
for (i in seq_len(nrow(cv))) {
  x <- folds[folds$candidate == cv$candidate[i], ]
  stopifnot(nrow(x) == 5, length(unique(x$fold)) == 5,
            abs(mean(x$Macro_F1) - cv$mean_macro_f1[i]) < 1e-12,
            abs(sd(x$Macro_F1) - cv$sd_macro_f1[i]) < 1e-12,
            abs(mean(x$Accuracy) - cv$mean_accuracy[i]) < 1e-12)
}
cm <- read("outputs/random_forest/confusion_matrix.csv")
classes <- read("outputs/random_forest/per_class_metrics.csv")
for (i in seq_len(nrow(classes))) {
  label <- classes$class[i]
  support <- sum(cm$Freq[cm$Observed == label])
  predicted <- sum(cm$Freq[cm$Predicted == label])
  tp <- sum(cm$Freq[cm$Observed == label & cm$Predicted == label])
  stopifnot(support == classes$support[i],
            abs(tp / support - classes$recall[i]) < 1e-12,
            abs(tp / predicted - classes$precision[i]) < 1e-12)
}
stopifnot(sum(cm$Freq) == 13497)
source("verify_selected_model.R")

for (path in c("reports/RF_Selected_Model_Preview.html",
               "reports/STAT5003_Final_Report.html")) {
  doc <- xml2::read_html(path)
  body <- gsub("[[:space:]]+", " ", xml2::xml_text(xml2::xml_find_first(doc, "//body")))
  required <- c("0.482", "0.487", "0.454", "29.0%", "8.1%", "2418",
                "3,194", "not a pristine independent confirmation")
  stopifnot(all(vapply(required, function(x) grepl(x, body, fixed = TRUE), logical(1))))
  blocks <- xml2::xml_text(xml2::xml_find_all(doc, "//pre"))
  stopifnot(any(grepl("ranger::ranger", blocks, fixed = TRUE)),
            any(grepl("type = \"class\"", blocks, fixed = TRUE)),
            length(xml2::xml_find_all(doc, "//table")) >= 4)
  scripts <- xml2::xml_text(xml2::xml_find_all(doc, "//script"))
  stopifnot(any(grepl('initializeCodeFolding("hide" === "show")', scripts, fixed = TRUE)))
}
cat("PASS: RF folds and class metrics, selected-model evidence, rendered tables, code and folding.\n")
