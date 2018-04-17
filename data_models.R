
library(nnet)
library(Metrics)

setwd('/Users/pierlim/R_Projects/KE5206NN')
train_df <- read.csv('fields_removed.csv')

# min max normalisation
max_share_val <- max(train_df$shares)
min_share_val <- min(train_df$shares)
maxs = apply(train_df, 2, max)
mins = apply(train_df, 2 ,min)
shares = as.numeric(train_df$shares)
train_scaled_df <- as.data.frame(scale(train_df, center = mins, scale = maxs - mins))

# Neural Nets + BP
nn_df <- train_scaled_df

set.seed(42)
cat("\nCreating and training a neural network . . \n")
mynn <- nnet(shares ~ ., data=nn_df, linout=TRUE,
             size=30,skip=TRUE, MaxNWts=10000, trace=FALSE, maxit=100)
pred <- predict(mynn)
cm <- as.data.frame(cbind(nn_df$shares, pred))

pred_shares <-  pred * (max_share_val - min_share_val) + min_share_val
actual_shares <- (nn_df$shares * (max_share_val - min_share_val)) + min_share_val
rmse_result <- rmse(actual_shares, pred_shares)
print("Neural Network RMSE:")
print(rmse_result)
plot(shares, pred, col='blue', pch=16, ylab="predicted shares", xlab="real shares")

# RBF Model
library(RSNNS)
nn_df <- train_scaled_df
set.seed(42)
cat("\nCreating and training a RBF network . . \n")
rbf_model <- rbf(nn_df, nn_df$shares, size=5, linOut=TRUE)
rbf_predict <- fitted(rbf_model)

pred_shares <-  rbf_predict * (max_share_val - min_share_val) + min_share_val
actual_shares <- (nn_df$shares * (max_share_val - min_share_val)) + min_share_val
rmse_result <- rmse(actual_shares, pred_shares) 
print("RBF Model RMSE:")
print(rmse_result)
# observation -> when size is increased, the predicted values become more and more similar

# GRNN Model
pred_grnn <- function(x, nn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i = xlst, .combine = rbind) %dopar% {
    data.frame(pred = guess(nn, as.matrix(i)), i, row.names = NULL)
  }
}

library(grnn)
library(doSNOW)
library(doParallel)
nn_df <- train_scaled_df
grnn <- smooth(learn(nn_df, variable.column = ncol(nn_df)), sigma=0.2)
pred <- pred_grnn(nn_df, grnn) # pick 10 rows

pred_shares <-  pred * (max_share_val - min_share_val) + min_share_val
actual_shares <- (nn_df$shares * (max_share_val - min_share_val)) + min_share_val
rmse_result <- rmse(actual_shares, pred_shares) 
print("GRNN Model RMSE:")
print(rmse_result)

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
# cv <- foreach(s = seq(0.2, 1, 0.05), .combine = rbind) %dopar% {
#   grnn <- smooth(learn(train_df, variable.column = ncol(train_df)), sigma = s)
#   pred <- pred_grnn(set2[, -ncol(set2)], grnn)
#   test.sse <- sum((set2[, ncol(set2)] - pred$pred)^2)
#   data.frame(s, sse = test.sse)
# }




# library(neuralnet)
# features<-names(train_df)
# f <- paste(features,collapse = ' + ')
# f <- paste('shares ~', f)
# f <- as.formula(f)
# n <- names(train_df)
# nn <- neuralnet(f, train_df, hidden=c(10,10), linear.output=T)
# predicted <- compute(nn, train_df[1:ncol(train_df)])
# rmse_result <- rmse(train_df$shares, predicted)
# 
# # min max normalisation
# maxs = apply(train_df, 2, max)
# mins = apply(train_df, 2 ,min)
# shares = as.numeric(train_df$shares)
# trainNN <- as.data.frame(scale(train_df, center = mins, scale = maxs - mins))
# features<-names(trainNN[,1:ncol(trainNN)])
# f <- paste(features,collapse = ' + ')
# f <- paste('shares ~', f)
# f <- as.formula(f)
# nn <- neuralnet(f, trainNN, hidden=c(10,10), linear.output=T)