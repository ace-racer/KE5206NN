
rm(list=ls())
setwd('C:/Users/pierl/OneDrive/Documents/R_Projects/KE5206NN')
train_df <- read.csv('./data/train_70.0.csv')

# combine data_channel_is_? and weekday_is_? fields
train_df$weekday <- paste(train_df$weekday_is_monday, train_df$weekday_is_tuesday, train_df$weekday_is_wednesday, train_df$weekday_is_thursday, 
                 train_df$weekday_is_friday, train_df$weekday_is_saturday, train_df$weekday_is_sunday, sep='')

train_df$data_channel_type <- paste(train_df$data_channel_is_bus, train_df$data_channel_is_entertainment, train_df$data_channel_is_lifestyle,
                                    train_df$data_channel_is_socmed, train_df$data_channel_is_tech, train_df$data_channel_is_world, sep='')

# remove data_channel_is_? and weekday_is_? fields
train_df <- train_df[,-15:-20] 
train_df <- train_df[,-27:-33]

write.csv(train_df, file = "train_processed.csv")