# Description: generic machine learning tests, following Tech with Tim.
# He made a virtual environment for his pip install using Anaconda
# Tutorial: https://www.youtube.com/watch?v=ujTCoH21GlA

# conda create -n tensor python=3.6
# activate tensor
# pip install tensorflow
# pip install keras

# Change the python interpreter to be the virtual environment instead of the system interpreter
# Get dataset from: https://archive.ics.uci.edu/ml/datasets/Student+Performance

#import tensorflow
#import keras
import pandas #as pd (data = pandas.read_csv("student-mat.csv", sep=";"))
import numpy #as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pandas.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#print(data.head())
print("------------------------------------")

predict = "G3"

x = numpy.array(data.drop([predict], 1)) # Doesn't have predict (G3) in the new dataset
y = numpy.array(data[predict])           # So G3 is the one that it is predicting

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])