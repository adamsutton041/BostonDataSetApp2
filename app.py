import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import plotly.express as px

st.title("BostonDataSetApp")
data = pd.read_csv('housing.csv')
st.header("Data Frame")
st.write(data)
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
st.sidebar.header("Customise axis")
x = st.sidebar.selectbox(label="x axis", options=["RM","LSTAT", "PTRATIO", "MEDV"], index=0)
y = st.sidebar.selectbox(label="y axis", options=["RM","LSTAT", "PTRATIO", "MEDV"], index=1)
fig = px.scatter(data, x, y)
st.plotly_chart(fig)



# Shuffle and split the data into training and testing subsets
testSplit = st.sidebar.slider(label = "test/train split", min_value = 0, max_value = 1, value = 0.2, step = 0.1)
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=testSplit, random_state = 42)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn import tree
import matplotlib.pyplot as plt

st.sidebar.header("Decision Tree Parameters")
in1 = st.sidebar.slider(label = "depth", min_value = 1, max_value = 7, value = 4, step = 1)
in2 = st.sidebar.slider(label = "nodes", min_value = 1, max_value = 128, value = 4, step = 1)

regressor = DecisionTreeRegressor(max_depth=in1, max_leaf_nodes=in2, random_state = 1)
regressor.fit(X_train, y_train)
fig = plt.figure(figsize=(25,20))
_=tree.plot_tree(regressor)

st.pyplot(fig)

cv = ShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0)
sizes, train_scores, test_scores = learning_curve(regressor, features, prices, cv = cv, scoring = 'r2')

st.write("This is the performance metric:{}".format(np.round(test_scores.mean(),2)))
st.header("Enter you parameters:")
in1 = st.number_input(label = "RM", min_value = 0, value = 10)
in2 = st.number_input(label = "LSTAT", min_value = 0, value = 10)
in3 = st.number_input(label = "PTRATIO", min_value = 0, value = 20)
b1 = st.button(label = "Calculate")
if b1 == True:
    X_custom = np.array([in1, in2, in3]).reshape(1, -1)
    [y_predict] = regressor.predict(X_custom)
    st.write("Your house price should be {}".format(np.round(y_predict), 2))

