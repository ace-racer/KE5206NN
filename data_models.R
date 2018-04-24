library(nnet)
library(Metrics)

setwd('C:/Users/pierl/OneDrive/Documents/R_Projects/KE5206NN')
train_df <- read.csv('fields_removed.csv')
test_df <- read.csv('test_fields_removed.csv')

# min max normalisation
max_share_val <- max(train_df$shares)
min_share_val <- min(train_df$shares)
maxs = apply(train_df, 2, max)
mins = apply(train_df, 2 ,min)
shares = as.numeric(train_df$shares)
train_scaled_df <- as.data.frame(scale(train_df, center = mins, scale = maxs - mins))
# scale test data using train data's params
test_scaled_df <- as.data.frame(scale(test_df, center = mins, scale = maxs - mins)) 

# fn to convert from normalized to actual share value
get_num_shares <- function(share_norm, min_share_val, max_share_val) {
  return (share_norm * (max_share_val - min_share_val) + min_share_val)
}



#### ---------------------------------- FFNN + BP ------------------------------------ ####
# h2o's early stopping parameters default to 
# stopping_rounds = 5
# stopping_metric = deviance
# stopping_tolerance = 0.001
library(h2o)

nn_df <- train_scaled_df
nn_test_df <- test_scaled_df
h20_df <- nn_df[ , -which(names(nn_df) %in% c("shares"))]
dl_fit1 <- h2o.deeplearning(x = names(h20_df),
                            y = "shares",
                            training_frame = as.h2o(nn_df),
                            model_id = "dl_fit1",
                            hidden = c(100, 100, 50),
                            nfolds = 3,
                            seed = 1, epochs = 10000, l1=1e-5, l2=1e-5, activation = "Maxout") 

h2o_predict <- as.data.frame(h2o.predict(dl_fit1, as.h2o(nn_test_df)))
pred_shares <- get_num_shares(h2o_predict, min_share_val, max_share_val)
actual_shares <- get_num_shares(nn_test_df$shares, min_share_val, max_share_val)
print("RMSE for neural network is :")
print(rmse(nn_test_df$shares, h2o_predict))
print("Training RMSE:")
print(h2o.rmse(dl_fit1))
plot(dl_fit1, timestep = "epochs", metric = "rmse")

h2o.init()
# random search
activation_opt <- c("Rectifier", "Maxout", "Tanh")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)

hyper_params <- list(activation = activation_opt, l1 = l1_opt, l2 = l2_opt)
#search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs = 600)
search_criteria <- list(strategy = "Cartesian")

splits <- h2o.splitFrame(as.h2o(nn_df), ratios = 0.8, seed = 1)
dl_grid <- h2o.grid("deeplearning", x = names(h20_df),
                    y = "shares",
                    grid_id = "dl_grid2",
                    training_frame = splits[[1]],
                    validation_frame = splits[[2]],
                    seed = 1,
                    hidden = c(100,100,50),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)

dl_gridperf <- h2o.getGrid(grid_id = "dl_grid2", 
                           sort_by = "rmse", 
                           decreasing = FALSE)
print(dl_gridperf)

#### ---------------------------------- RBF ------------------------------------ ####
# RBF Model : observation -> when size is increased, the predicted values become more and more similar
library(RSNNS)
nn_df <- train_scaled_df
nn_test_df <- test_scaled_df
set.seed(42)
cat("\nCreating and training a RBF network . . \n")
rbf_model <- rbf(nn_df, nn_df$shares, size=8, linOut=TRUE,
                 initFunc = "RBF_Weights", initFuncParams = c(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                 learnFunc = "RadialBasisLearning", learnFuncParams = c(1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05,1e-05))
rbf_predict <- predict(rbf_model, nn_test_df)

pred_shares <- get_num_shares(rbf_predict, min_share_val, max_share_val)
actual_shares <- get_num_shares(nn_test_df$shares, min_share_val, max_share_val)
rmse_result <- rmse(nn_test_df$shares, rbf_predict) 
print("RBF Model RMSE:")
print(rmse_result)

# Home made grid search for max_size
max_size <- 50
best_rmse <- 999.0
best_rbf_model <- NA
for (i in 2:max_size) {   # putting size=1 will crash rbf
  rbf_model <- rbf(nn_df, nn_df$shares, size=i, linOut=TRUE)
  rbf_predict <- predict(rbf_model, nn_test_df)
  rmse_result <- rmse(nn_test_df$shares, rbf_predict)
  if (rmse_result < best_rmse) {
    best_rmse <- rmse_result
    best_rbf_model <- rbf_model
    cat("best size param = ", i, "\n")
  }
}

#### ---------------------------------- GRNN ------------------------------------ ####
# omg, it runs forever! 

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
nn_test_df <- test_scaled_df
grnn <- smooth(learn(nn_df, variable.column = ncol(nn_df)), sigma=0.2)
pred <- pred_grnn(nn_test_df[1:100,], grnn) # cut out 100, else it will run forever

pred_shares <-  pred * (max_share_val - min_share_val) + min_share_val
actual_shares <- (nn_df$shares * (max_share_val - min_share_val)) + min_share_val
rmse_result <- rmse(nn_df$shares, pred) 
print("GRNN Model RMSE:")
print(rmse_result)

# PNN was not tried because it is used for classification problems

