
library(nnet)
library(Metrics)

setwd('C:/Users/pierl/OneDrive/Documents/R_Projects/KE5206NN')
train_df <- read.csv('fields_removed.csv')

# min max normalisation
max_share_val <- max(train_df$shares)
min_share_val <- min(train_df$shares)
maxs = apply(train_df, 2, max)
mins = apply(train_df, 2 ,min)
shares = as.numeric(train_df$shares)
nn_df <- as.data.frame(scale(train_df, center = mins, scale = maxs - mins))

set.seed(42)
cat("\nCreating and training a neural network . . \n")
mynn <- nnet(shares ~ ., data=nn_df, linout=TRUE,
             size=50,skip=TRUE, MaxNWts=10000, trace=FALSE, maxit=100)
pred <- predict(mynn)
cm <- as.data.frame(cbind(nn_df$shares, pred))

pred <-  pred * (max_share_val - min_share_val) + min_share_val
rmse_result <- rmse(nn_df$shares, pred)
shares <- (nn_df$shares * (max_share_val - min_share_val)) + min_share_val
plot(shares, pred, col='blue', pch=16, ylab="predicted shares", xlab="real shares")


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