setwd("~/Desktop/fall-2019-stats-101c")

library(dplyr)
library(lubridate)
library(caret)
library(tidyverse)
library(glmnet)
library(rpart)
library(caretEnsemble)
library(mlbench)
library(randomForest)
library(nnet)

set.seed(2020)

train_156 <- read.csv("train_156.csv",stringsAsFactors = TRUE)[-1]
test_155 <- read.csv("test_155.csv",stringsAsFactors = TRUE)[-1]
train_156 %>% arrange(gameID)
train_156$date <- as.character(train_156$date)
train_156 <- train_156 %>% mutate(date = ymd(date)) %>% mutate_at(vars(date), funs(year))
train_156$date <- as.factor(train_156$date)
lvdate <- levels(train_156$date)
train_156<-train_156 %>%  mutate(HTWins = as.factor(ifelse(train_156$HTWins == 0, "No", "Yes")))


## Train-test split according to years
train=train_156[as.numeric(as.character(train_156$date))<=2010,]
test=train_156[as.numeric(as.character(train_156$date))>2010,]

## Caret Ensemble
time_slices <- createTimeSlices(1:nrow(train),initialWindow =1600,horizon =400, 
                                fixedWindow = FALSE)
#str(time_slices)
myTimeControl <- trainControl(method = "cv", number = 2,  
                              savePrediction = TRUE,classProbs = TRUE,
                              returnResamp = "final",returnData = TRUE,
                              index= time_slices$train, indexOut=time_slices$test )

#repeats = 1,

model_list_big <- caretList(
  HTWins~HT.S1.pts+VT.OS1.plmin+VT.OS3.plmin+
    VT.OTA.ast+VT.OTS.pts+VT.S1.plmin+VT.S2.plmin+
    VT.OS2.plmin+VT.OS4.plmin+VT.OTA.dreb+VT.S3.plmin+
    VT.S4.plmin+HT.OS3.dreb+VT.OS4.dreb+VT.OS5.plmin+
    VT.OTS.fgm+VT.TA.ast+VT.TA.pts+VT.TS.pts+HT+
    HT.OS1.dreb+HT.OS4.oreb+HT.S3.pts+VT.OS1.oreb+
    VT.S5.pts+VT.S5.stl, data=train,
  trControl=myTimeControl,
  metric='Accuracy',
  methodList=c('rf', 'gbm','treebag', 'nodeHarvest','glm','rpart'),
  tuneList=list(
    rf=caretModelSpec(method='rf',tunelength = 10, ntrees = 1000, importance = TRUE),
    gbm=caretModelSpec(method='gbm',tuneGrid= expand.grid(.interaction.depth = seq(1, 7, by = 2), 
                                                          .n.trees = seq(300,1200,by = 50), 
                                                          .shrinkage = c(0.01, 0.1) ) ),
    tbag=caretModelSpec(method='treebag'),
    Nharvest=caretModelSpec(method='nodeHarvest',nodes = 100),
    rf1=caretModelSpec(method="rf", tuneGrid=data.frame(.mtry=5)),
    rf2=caretModelSpec(method="rf", tuneGrid=data.frame(.mtry=10), preProcess="pca"),
    nn=caretModelSpec(method="nnet", tuneLength=2, trace=FALSE)
  )
)

greedy_ensemble <- caretEnsemble(model_list_big)
summary(greedy_ensemble)

ens_preds <- predict(greedy_ensemble, newdata=test)
ens_sub <- predict(greedy_ensemble, newdata=test_156)

testpreds <- data.frame(test$id, ens_preds)
subpreds <- data.frame(test_155$id, ens_sub)
colnames(testpreds) <- c("id", "HTWins")
colnames(subpreds) <- c("id", "HTWins")
cat("saving the submission file\n")
write.csv(subpreds, "submission_caretem.csv", row.names = FALSE)
write.csv(testpreds, "testpreds_caretem.csv", row.names = FALSE)