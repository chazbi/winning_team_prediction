---
  title: "101c_feature_selection"
output: html_document
---
  
## load packages

library(dplyr)
library(tidyr)
library(purrr)
library(zoo)
library(corrplot)
library(ggcorrplot)
library(BBmisc)
library(MLmetrics)
library(olsrr)
library(glmnet)
library(caret)
library(randomForest)
library(xgboost)
library(mlr)
library(factoextra)
library(pracma)
library(caTools)
library(TH.data)

## load data

train <- read.csv("train_156.csv", stringsAsFactors = FALSE)
train <- train %>% dplyr::select(-X)
test <- read.csv("test_155.csv", stringsAsFactors = FALSE)
test  <- test %>% dplyr::select(-X)
head(train);head(test)

#data cleaning
train$VT <- as.factor(train$VT)
train$HT <- as.factor(train$HT)
train$HTWins <- as.factor(train$HTWins)
train$VTleague <- as.factor(train$VTleague)
train$HTleague <- as.factor(train$HTleague)

test$VT <- as.factor(test$VT)
test$HT <- as.factor(test$HT)
test$VTleague <- as.factor(test$VTleague)
test$HTleague <- as.factor(test$HTleague)


## feature selection
### 1. Boruta

library(Boruta)
boruta_output <- Boruta(HTWins ~ ., data=train, doTrace=0)  
names(boruta_output)

# Get significant variables including tentatives
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  

# Do a tentative rough fix
roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

# Variable Importance Scores
imps <- attStats(roughFixMod)
imps2 <- imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ], 30)  # descending sort

# Plot variable importance
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")


### 2. ML Algorithms

# Train an rpart model and compute variable importance.
library(caret)
set.seed(10000)
rPartMod <- caret::train(HTWins ~ ., data=train, method="rpart")
rpartImp <- varImp(rPartMod)
print(rpartImp)

# Train an RRF model and compute variable importance.
set.seed(100)
rrfMod <- caret::train(HTWins ~ ., data=train, method="RRF")
rrfImp <- varImp(rrfMod, scale=F)
rrfImp
plot(rrfImp, top = 20, main='Variable Importance')


### 3. LASSO regression

library(glmnet)
train_x <- as.matrix(train[, -c(1:3, 6:7)])
train_y <- as.vector(train$HTWins)

# Fit the LASSO model (Lasso: Alpha = 1)
set.seed(100)
cv.lasso <- cv.glmnet(train_x, train_y, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')

# Results
plot(cv.lasso)

# plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
cat('Min Lambda: ', cv.lasso$lambda.min, '\n 1Sd Lambda: ', cv.lasso$lambda.1se)
df_coef <- round(as.matrix(coef(cv.lasso, s=cv.lasso$lambda.min)), 2)

# See all contributing variables
df_coef[df_coef[, 1] != 0, ]


### 4. step-wise selection (backward & forward)

# Step 1: Define base intercept only model
train$HTWins <- as.numeric(train$HTWins)
base.mod <- lm(HTWins ~ 1 , data=train)  

# Step 2: Full model with all predictors
all.mod <- lm(HTWins ~ . , data= train) 

# Step 3: Perform step-wise algorithm. direction='both' implies both forward and backward stepwise
stepMod <- step(base.mod, scope = list(lower = base.mod, upper = all.mod), direction = "both", trace = 0, steps = 1000)  

# Step 4: Get the shortlisted variable.
shortlistedVars <- names(unlist(stepMod[[1]])) 
shortlistedVars <- shortlistedVars[!shortlistedVars %in% "(Intercept)"] # remove intercept

# Show
print(shortlistedVars)


### 5. recursive feature elimanation (RFE)

set.seed(100)
options(warn=-1)

subsets <- c(15:20, 25, 50, 76)
subsets <- 25

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

train$HTWins <- as.factor(train$HTWins)  #change to factor

lmProfile <- rfe(x= train[, -c(1:3, 6:7)], y=train$HTWins,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfil
