library(tidyverse)

data <- read.csv('train.csv')
data$Y <- as.factor(data$Y)

X = scale(data[,1:10])

#covariance matrix CX
CX = cov(X)
head(CX)

eigenX = eigen(CX, symmetric = TRUE)

#scree plot
plot(eigenX$value[1:10], type = "b", xlab = "principal component", ylab = "eigenvalue", col = "green")
eigenX$value[1:10]

#transpose eigen vectors so that row one is eigen vector 1
eigenRow = t(eigenX$vectors)

#find principal component 1 values for all 569 samples
PC1 = matrix(NA, ncol = 1, nrow = 100000)

for (l in 1:100000){
  PC1[l,1] = 0
}

for (i in 1:100000){
  for (j in 1:10){
    PC1[i,1] = PC1[i,1] + eigenRow[1,j] * X[i,j]
  }
}

#find principal component 2 values for all 569 samples
PC2 = matrix(NA, ncol = 1, nrow = 100000)

for (k in 1:100000){
  PC2[k,1] = 0
}

for (i in 1:100000){
  for (j in 1:10){
    PC2[i,1] = PC2[i,1] + eigenRow[2,j] * X[i,j]
  }
}

#plot first and second principal components 
data$PC1 <-PC1
data$PC2 <-PC2
ggplot(data = data, aes(x = PC1, y = PC2, color = Y)) + 
         geom_point() + 
         ggtitle("Scatter Plot of Principal Components 1 and 2") 