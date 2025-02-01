import numpy as np
import pandas as pd
import os

# check the os directory 
print("Current Working Directory:", os.getcwd())
#to set the os directory and checking the data
os.chdir("/Users/thaqiyuddin/Personal/thaqi/master/advancedai/Advanced_AI/Practice/Data Preprocessing")
dataset = pd.read_csv("Data.csv")
print('Load the datasets...')

#print shape to know number of row and column
print ('dataset: %s'%(str(dataset.shape)))


#here is the data splitting since all the number of column is 4
#so x will be taking all accept the last column and the Y will take take the column number 4 only
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#to prove all the above
print ('X: %s'%(str(X)))
print ('-----------------------------------')
print ('Y: %s'%(str(Y)))

#to replace missing value
from sklearn.impute import SimpleImputer  # Updated import for handling missing values

# Replace missing values using SimpleImputer
# Parameters:
# - missing_values: the placeholder for missing data (e.g., np.nan)
# - strategy: how to fill missing values ('mean', 'median', or 'most_frequent')
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on columns 1 and 2 of X (index 0-based)
imputer = imputer.fit(X[:, 1:3])

# Replace missing data with the mean of each column
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Display the updated X
print('Updated X:', X)



#to lable the country
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encode the first column of X (categorical feature)
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Apply one-hot encoding to the first column
# ColumnTransformer is now used to apply transformations to specific columns
ct = ColumnTransformer(
    [("onehot", OneHotEncoder(), [0])],  # Apply OneHotEncoder to column index 0
    remainder="passthrough"  # Leave the other columns untouched
)
X = ct.fit_transform(X)

# Output the result
print("Encoded X:")
print(X)


