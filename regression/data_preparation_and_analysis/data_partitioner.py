import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Update the working directory
os.chdir("H:\\KE5206CA1\\code\\regression\\data_preparation_and_analysis")

# Split ratios
training_split = 0.7
testing_split = 0.3

should_split_only_test = True

input_df = pd.read_csv("../data/OnlineNewsPopularity.csv")
training_df, testing_df = train_test_split(input_df, test_size=testing_split, random_state=42)
training_df.to_csv("../data/train_"+str(training_split * 100) + ".csv")

if not should_split_only_test:
    testing_split = testing_split / 2
    testing_df, validation_df = train_test_split(testing_df, test_size=testing_split, random_state=42)
    validation_df.to_csv("../data/validation_"+str(training_split * 100) + ".csv")

testing_df.to_csv("../data/test_"+str(testing_split * 100) + ".csv")