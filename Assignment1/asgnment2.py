## this assignment is created by Thaqiyuddin bin Mizan
## KCA24006
## Advanced AI Assignment 2 

import pandas as pd

file_path = "KCA24006_CarPrice_Assignment.csv"  
df = pd.read_csv(file_path)
numerical_columns = df.select_dtypes(include=['number']).columns
df_numerical = df[numerical_columns]

# normalization part
df_minmax_0_1 = (df_numerical - df_numerical.min()) / (df_numerical.max() - df_numerical.min())
df_minmax_neg1_1 = 2 * ((df_numerical - df_numerical.min()) / (df_numerical.max() - df_numerical.min())) - 1

# Save the results as CSV files
df_minmax_0_1.to_csv("KCA24006_CarPrice_Normalized_0_1.csv", index=False)
df_minmax_neg1_1.to_csv("KCA24006_CarPrice_Normalized_neg1_1.csv", index=False)

#show output in terminal when all is finished
print("Normalization completed. Files saved as:")
print("1. KCA24006_CarPrice_Normalized_0_1.csv")
print("2. KCA24006_CarPrice_Normalized_neg1_1.csv")