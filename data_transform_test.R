# We skip outlier removal for the data_transformation of test data

rm(list=ls())
#setwd('C:/Users/pierl/OneDrive/Documents/R_Projects/KE5206NN')
setwd('/Users/pierlim/R_Projects/KE5206NN')
test_df <- read.csv('./data/test_30.0.csv')

# combine data_channel_is_? and weekday_is_? fields
test_df$weekday <- paste(test_df$weekday_is_monday, test_df$weekday_is_tuesday, test_df$weekday_is_wednesday, test_df$weekday_is_thursday, 
                          test_df$weekday_is_friday, test_df$weekday_is_saturday, test_df$weekday_is_sunday, sep='')

test_df$data_channel_type <- paste(test_df$data_channel_is_bus, test_df$data_channel_is_entertainment, test_df$data_channel_is_lifestyle,
                                    test_df$data_channel_is_socmed, test_df$data_channel_is_tech, test_df$data_channel_is_world, sep='')

# remove data_channel_is_? and weekday_is_? fields
test_df <- test_df[,-15:-20] 
test_df <- test_df[,-27:-33]


library(corrplot, quietly=TRUE)

# Correlations work for numeric variables only.
numeric_variables   <- c("timedelta", "n_tokens_title", "n_tokens_content",
                         "n_unique_tokens", "n_non_stop_words",
                         "n_non_stop_unique_tokens", "num_hrefs", "num_self_hrefs",
                         "num_imgs", "num_videos", "average_token_length",
                         "num_keywords", "kw_min_min", "kw_max_min", "kw_avg_min",
                         "kw_min_max", "kw_max_max", "kw_avg_max", "kw_min_avg",
                         "kw_max_avg", "kw_avg_avg", "self_reference_min_shares",
                         "self_reference_max_shares", "self_reference_avg_sharess",
                         "is_weekend", "LDA_00", "LDA_01", "LDA_02", "LDA_03",
                         "LDA_04", "global_subjectivity",
                         "global_sentiment_polarity", "global_rate_positive_words",
                         "global_rate_negative_words", "rate_positive_words",
                         "rate_negative_words", "avg_positive_polarity",
                         "min_positive_polarity", "max_positive_polarity",
                         "avg_negative_polarity", "min_negative_polarity",
                         "max_negative_polarity", "title_subjectivity",
                         "title_sentiment_polarity", "abs_title_subjectivity",
                         "abs_title_sentiment_polarity")

correlations <- cor(test_df[, numeric_variables], use="pairwise", method="pearson")

# Order the correlations by their strength.
correlations_ordered <- order(correlations[1,])
correlations <- correlations[correlations_ordered, correlations_ordered]

# Display the actual correlations.
print(correlations)

# Graphically display the correlations.
corrplot(correlations, mar=c(0,0,1,0))
title(main="Correlation train_processed.csv using Pearson",
      sub=paste("Rattle", format(Sys.time(), "%Y-%b-%d %H:%M:%S"), Sys.info()["user"]))
print(ncol(test_df))

test_df <- test_df[ , -which(names(test_df) %in% c("n_non_stop_words","n_unique_tokens.", "kw_max_min.", "self_reference_max_shares", "self_reference_min_shares",
                                                      "kw_max_avg", "max_positive_polarity", "rate_positive_words", "min_negative_polarity", "rate_negative_words", "url"))]
print(ncol(test_df))

write.csv(test_df, file = "test_fields_removed.csv")