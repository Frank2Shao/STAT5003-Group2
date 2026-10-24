required_packages <- c(
  "readr",
  "dplyr",
  "caret",
  "recipes",
  "ggplot2"
)

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
library(caret)
library(recipes)
library(ggplot2)

if (!exists("scaled_recipe")) {source("06_model_setup.R")}

# Training Model

lda_recipe <- scaled_recipe %>% # Additional pre-processing
  step_lincomb(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.9)

lda_fit <- train(
  lda_recipe,
  data = training_model_data,
  trControl = model_control,
  method = "lda2",
  metric = "Macro_F1",
  tuneGrid = expand.grid(dimen=1:2), # Only 2 possible values for hyperparam tuning
  prior = rep(1/3, 3) # To account for class imbalance in the dataset 
)

lda_fit$resample
lda_fit$results
lda_fit$bestTune

# Performance on data

## Test Data
lda.preds<- predict(lda_fit, newdata=test_model_data)
lda.confusion <- confusionMatrix(data=lda.preds, reference=test_model_data$rating_tier)
lda.confusion

## Training Data (Out of Fold)
lda.train.confusion <- confusionMatrix(data=lda_fit$pred$pred, reference=lda_fit$pred$obs)
lda.train.confusion

# Plots

lda.confusion.df <- as.data.frame(lda.confusion$table) %>%
  group_by(Reference) %>%
  mutate(
    row_pct=100*Freq/sum(Freq)
  ) %>%
  ungroup()

lda.test.plot <- ggplot(lda.confusion.df, aes(x=Prediction, y=Reference, fill=row_pct)) +
  geom_tile(colour="white")+
  geom_text(
    aes(label=paste0(Freq, "\n(", round(row_pct,1), "%)")),
    size=4
  )+
  scale_fill_gradient(low="white", high="steelblue", name="% of true class")+
  scale_y_discrete(limits=rev(levels(lda.confusion.df$Reference)))+
  labs(
    title = "LDA Predictions on Test Set",
    x = "Predicted Rating",
    y = "Actual Rating"
  )+
  coord_fixed()+
  theme_minimal()

lda.train.confusion.df <- as.data.frame(lda.train.confusion$table) %>%
  group_by(Reference) %>%
  mutate(
    row_pct=100*Freq/sum(Freq)
  ) %>%
  ungroup()

lda.train.plot <- ggplot(lda.train.confusion.df, aes(x=Prediction, y=Reference, fill=row_pct)) +
  geom_tile(colour="white")+
  geom_text(
    aes(label=paste0(Freq, "\n(", round(row_pct,1), "%)")),
    size=4
  )+
  scale_fill_gradient(low="white", high="steelblue", name="% of true class")+
  scale_y_discrete(limits=rev(levels(lda.confusion.df$Reference)))+
  labs(
    title = "LDA Predictions on Training Set",
    x = "Predicted Rating",
    y = "Actual Rating"
  )+
  coord_fixed()+
  theme_minimal()

ggsave("figures/LDA_train_cm.png", lda.train.plot, width = 6, height = 6, dpi = 220)
ggsave("figures/LDA_test_cm.png" , lda.test.plot , width = 6, height = 6, dpi = 220)