### Time Series Ridge

## Reading Libraries
library(dplyr)
library(lubridate)
library(caret)
library(tidyverse)
library(glmnet)
library(leaps)

set.seed(2020)

## Reading Data
train_156 <- read.csv("train_156.csv",stringsAsFactors = TRUE)[-1]
test_155 <- read.csv("test_155.csv",stringsAsFactors = TRUE)[-1]
train_156 %>% arrange(gameID)
train_156$date <- as.character(train_156$date)
train_156 <- train_156 %>% mutate(date = ymd(date)) %>% mutate_at(vars(date), funs(year))
train_156$date <- as.factor(train_156$date)
lvdate <- levels(train_156$date)
train_156<-train_156 %>%  mutate(HTWins = as.factor(ifelse(train_156$HTWins == 0, "No", "Yes")))

## Test-train Split
idx = sample(length(train_156$HTWins), 0.75*length(train_156$HTWins))
train = train_156[idx, ]
test = train_156[-idx, ]

y_train <- train$HTWins
y_test <- test$HTWins
x_train <- model.matrix(HTWins~HT.S1.pts+VT.OS1.plmin+VT.OS3.plmin+
                          VT.OTA.ast+VT.OTS.pts+VT.S1.plmin+VT.S2.plmin+
                          VT.OS2.plmin+VT.OS4.plmin+VT.OTA.dreb+VT.S3.plmin+
                          VT.S4.plmin+HT.OS3.dreb+VT.OS4.dreb+VT.OS5.plmin+
                          VT.OTS.fgm+VT.TA.ast+VT.TA.pts+VT.TS.pts+HT+
                          HT.OS1.dreb+HT.OS4.oreb+HT.S3.pts+VT.OS1.oreb+
                          VT.S5.pts+VT.S5.stl,train )
x_test <- model.matrix(HTWins~HT.S1.pts+VT.OS1.plmin+VT.OS3.plmin+
                         VT.OTA.ast+VT.OTS.pts+VT.S1.plmin+VT.S2.plmin+
                         VT.OS2.plmin+VT.OS4.plmin+VT.OTA.dreb+VT.S3.plmin+
                         VT.S4.plmin+HT.OS3.dreb+VT.OS4.dreb+VT.OS5.plmin+
                         VT.OTS.fgm+VT.TA.ast+VT.TA.pts+VT.TS.pts+HT+
                         HT.OS1.dreb+HT.OS4.oreb+HT.S3.pts+VT.OS1.oreb+
                         VT.S5.pts+VT.S5.stl,test )
x_submit<-model.matrix(~HT.S1.pts+VT.OS1.plmin+VT.OS3.plmin+
                         VT.OTA.ast+VT.OTS.pts+VT.S1.plmin+VT.S2.plmin+
                         VT.OS2.plmin+VT.OS4.plmin+VT.OTA.dreb+VT.S3.plmin+
                         VT.S4.plmin+HT.OS3.dreb+VT.OS4.dreb+VT.OS5.plmin+
                         VT.OTS.fgm+VT.TA.ast+VT.TA.pts+VT.TS.pts+HT+
                         HT.OS1.dreb+HT.OS4.oreb+HT.S3.pts+VT.OS1.oreb+
                         VT.S5.pts+VT.S5.stl,test_155)


## Model Building

## Validation Setting with Timeslices
myTimeControl <- trainControl(method = "timeslice",
                              initialWindow = 1200,
                              horizon = 800,
                              fixedWindow = FALSE)

## Fit Model, tune parameter
ridge_fit <- train(x = x_train, y = y_train, 
                   method = 'glmnet',
                   trControl = myTimeControl,
                   tuneGrid = expand.grid(alpha = seq(0, 1, 0.1),
                                          lambda = seq(0, 1, 0.01)))

## test info
ridge_info <- postResample(predict(ridge_fit, x_test), y_test)
ridge_info

## Submission File Generation
ridge_sub <- predict(ridge_fit, x_submit)
ridge_preds <- data.frame(test_155$id, ridge_sub)
colnames(ridge_preds) <- c("id", "HTWins")
cat("saving the submission file\n")
write.csv(ridge_preds, "submission_ridge.csv", row.names = FALSE)