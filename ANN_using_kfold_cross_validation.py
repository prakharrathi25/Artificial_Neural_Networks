''' Importing Necessary Libraries '''

# Data Wrangling 
import pandas as pd
import numpy as np 

# Data Visualisation 
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Tools
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Deep Learning Libraries
 import tensorflow as tf
import keras 
import theano

# Model Building Tools from Keras Library
from keras.models import Sequential 
from keras.layers import Dense


''' Read Data into Pandas Dataframe '''

data = pd.read_csv('Churn_Modelling.csv', index_col = 'RowNumber')
print(data.shape)
print(data.head())

'''Data Preprocessing'''
data.isnull().sum() # No null values

# Encoding Categorical Data

# Label encode gender 
data['Gender'] = data['Gender'].map( {'Female' : 1, 'Male': 0} ) 
data.head(10)

# Making dummies for geography
data[['France', 'Germany', 'Spain',]] = pd.get_dummies(data['Geography']) 
data.drop(['Geography'], axis = 1, inplace = True)
data.head(10)

# Drop one dummy variable to avoid
data.drop(['Germany'], axis = 1, inplace = True)

# Visualizing correlations 
plt.figure(figsize = (15, 6))
sns.heatmap(data.corr(), vmax = 0.5, annot = True, cmap = 'RdYlGn')

# Creating features and labels 
X = data.drop(['Exited', 'CustomerId', 'Surname'], axis = 1).values
y = data['Exited'].values
print(X, y)

# Split the data into test and train set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

''' Building the Artificial Neural Network '''

from keras.models import Sequential 
from keras.layers import Dense

# Evaluating our ANN using K-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# Building the function which returns our ANN 
def  build_classifier():
    classifier = Sequential()
    classifier.add(layer = Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11 ))
    classifier.add(layer = Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu' ))
    classifier.add(layer = Dense(units = 1, kernel_initializer = 'uniform' , activation = 'sigmoid')) 
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier 

# Building our global classifier
Classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# Building the K-Fold cross val function
accuracies = cross_val_score(estimator = Classifier, X = X_train, y = y_train, cv = 10)

# Calculating the accuracies
print(accuracies)
print('Mean is:'. format(accuracies.mean()))
print('Standard Deviation is:'. format(accuracies.std()))
	