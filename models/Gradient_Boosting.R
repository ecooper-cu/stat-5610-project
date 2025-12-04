#install.packages("MLmetrics")
library(xgboost)
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(ipred)
library(MLmetrics)

data <- read.csv('train.csv')
trainX <- as.matrix(data[,1:10])
trainY <- factor(data$Y, levels = c(0, 1), labels = c("Class_0", "Class_1"))

control <- trainControl(
  method = "cv",            
  number = 10,               
  verboseIter = TRUE,       
  returnData = FALSE,
  returnResamp = "all",
  classProbs = TRUE,        
  summaryFunction = prSummary, 
  allowParallel = TRUE      
)

param_grid <- expand.grid(
  #number of trees created
  nrounds = c(400,500,750,1000),
  
  #maximum depth of each tree
  max_depth = c(3, 5, 7, 9),
  
  #shrinkage parameter/learning rate
  eta = c(0.001, 0.005, 0.01),
  
  #minimum loss reduction required to make a further partition on a leaf
  gamma = c(0,0.01),
  
  #subsample ratio of columns
  colsample_bytree = c(1.0),
  
  #minimum sum of instance weight needed in a child/ minimum number of instances needed to be in each node
  min_child_weight = c(1,3),
  
  #subsample ratio of training samples
  subsample = c(1.0)
)

xgb_model <- train(
  x = trainX,
  y = trainY,
  trControl = control,
  tuneGrid = param_grid,
  method = "xgbTree",
  metric = "F"
)

# Print the best model
print(xgb_model)

# Access the best tune
xgb_model$bestTune

# Plot the results (optional)
plot(xgb_model)


