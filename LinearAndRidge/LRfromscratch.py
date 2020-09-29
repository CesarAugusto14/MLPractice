import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Reading the Data:
df = pd.read_excel('Residential-Building-Data-Set.xlsx')
new_header = df.iloc[0]
df = df[1:]
df.columns = new_header
y_data = df[['V-10']]  # Response Variables (Target)
# Predictors
x_data = df.drop(columns=['START YEAR', 'START QUARTER',
                          'COMPLETION YEAR', 'COMPLETION QUARTER', 'V-9', 'V-10'])

# Splitting randomly the data to perform the test. We set random_state = 2, for reproducible output.
xtrain, xtest, ytrain, ytest = train_test_split(x_data, y_data, test_size=0.20, random_state=2)
print('The Splitting of the Data is Random and it is:\n ')
print("Test Samples:", xtest.shape[0])
print("Training Samples:", xtrain.shape[0])

# Deploying the Model:
# Both xtrain and xtest dtype= 'object', so:
xtrain = np.array(xtrain, dtype='float')
xtest = np.array(xtest, dtype='float')

# Array whose elements are all one. Same length as xtest.
onetest = np.ones(len(xtest))
# Array whose elements are all one. Same length as xtrain.
onetrain = np.ones(len(xtrain))
# xtrain with the vectore onetrain added as last column.
xtrainad = np.c_[xtrain, onetrain]
# xtest with the vector onetest added as last column.
xtestad = np.c_[xtest, onetest]
# Performing the Linear Regression:
b = np.dot(np.linalg.pinv(xtrainad), ytrain)        # Bhat.
yhat = np.dot(xtestad, b)                           # Fitted values of y.
yhat = np.array(yhat, dtype='float')
ytest = np.array(ytest, dtype='float')
# Metrics for this Model:
MAE = np.mean(np.abs(ytest-yhat))
correlation = np.corrcoef(yhat.T, ytest.T)
R2 = r2_score(yhat, ytest)
# Printing the metrics:
print('\n In this Part, The Selected Model is the Ordinary Linear Regression: ')
print('The Mean Absolute Error is: %.4f' % MAE)
print('The Coefficient of Correlation is: %.4f' % correlation[0][1])
print('The coefficient of Determination is: %.4f \n' % R2)

# Performing the Ridge Regression:
alpha = 0.1
# Tuning Parameter*Identity matrix.
k = alpha*np.eye(xtrainad.shape[1])
bridge = np.dot(np.dot(np.linalg.inv(
    np.dot(xtrainad.T, xtrainad)+k), xtrainad.T), ytrain)
yridge = np.dot(xtestad, bridge)
yridge = np.array(yridge, dtype='float')
# Metrics for this Model:
MAEridge = np.mean(np.abs(ytest-yridge))
correlation_ridge = np.corrcoef(yridge.T, ytest.T)
R2ridge = r2_score(yridge, ytest)
# Printing the Metrics:
print('\n In this Part, The Selected Model is the Ridge Regression, whit alpha = %0.2f,' % alpha)
print('The Mean Absolute Error is: %.4f' % MAE)
print('The Coefficient of Correlation is: %.4f' % correlation_ridge[0][1])
print('The coefficient of Determination is: %.4f \n' % R2ridge)
