#install.packages("randomForest")
library(randomForest)
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(ipred)

#read data, set seed
data <- read.csv('train.csv')
set.seed(1)

#to use one dev set instead of cross-val
#train <- sample_n(data, 75000) # picks 75% of the data
#dev <- anti_join(data, train)

#set m vales and n values to explore
m_vals = c(2,3,4,5)
n_trees = c(250,300,350,400,500)

#create folds
k_folds = createFolds(y = data$Y, k = 5, list = TRUE, returnTrain = FALSE)

#create F1 grid to store scores
F1_grid <- data.frame(m = c(0), n = c(0), F1 = c(0))
i = 1

#K-fold Cross Validation Loop
#for each m value explore each n value and calculate average F1 accross folds
for (m in m_vals){
  for (n in n_trees){
    #reset F1
    F1 = 0
    for (k in 1:5){
      #get dev set from fold and train set from remaining folds
      dev_indices <- k_folds[[k]]
      train <- data[-dev_indices, ]
      dev <- data[dev_indices, ]
      
      #generate rf model
      rf <- randomForest(as.factor(Y) ~ ., data = train, ntree = n, mtry = m)
      
      #make predictions on dev fold
      predictions <- predict(rf, newdata = dev, type = "class")
      
      #calc confusion matrix and F1
      conf_matrix <- confusionMatrix(predictions, as.factor(dev$Y), positive = "1")
      
      recall <- conf_matrix$byClass["Sensitivity"]
      precision <- conf_matrix$byClass["Pos Pred Value"]
      
      F1 = F1 + (2 * precision * recall) / (precision + recall)
    }
    
    #store average F1
    i = i + 1
    F1_grid[i,] <- list(m, n, F1/5)
  }
}

#print F1 grid
F1_grid

#set optimum tree
rf_opt <- randomForest(as.factor(Y) ~ ., data = train, ntree = 500, mtry = 4)

#read test data
test <- read.csv('test.csv')

#predict for test data
predictions <- predict(rf_opt, newdata = test, type = "class")

#prep for submission
RF_submission = data.frame(ID = test$ID, Y = predictions)
RF_submission$Y <- as.integer(RF_submission$Y)

j = 1
for (i in RF_submission$Y){
  if (i == 1){
    RF_submission[j,2]<- 0
  }
  else{
    RF_submission[j,2]<- 1
  }
  j = j+1
}

#store submission
file_path <- "/Users/Caroline/Desktop/StatLearningProject/RF_submission.csv"
write.csv(RF_submission, file=file_path, row.names = FALSE)
