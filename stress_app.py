
import numpy as np
import pandas as pd
import streamlit as st



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split    
from sklearn.metrics import confusion_matrix            
from sklearn.metrics import classification_report       # for detailed classification metrics
from sklearn.linear_model import LogisticRegression    
from sklearn.ensemble import RandomForestClassifier    
from sklearn.tree import DecisionTreeClassifier        
from sklearn.svm import SVC                            
from sklearn.neighbors import KNeighborsClassifier    




df = pd.read_csv("data/SaYoPillow.csv")
df.head()


X = df.drop(['sl'], axis=1)
y = df['sl']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# decision tree classifier
decision_tree_cl = DecisionTreeClassifier(criterion="gini", splitter="best")
decision_tree_cl.fit(x_train, y_train)
y_predict = decision_tree_cl.predict(x_test)


st.write("Human Stress Detection in and through Sleep")
st.divider()

number = st.number_input("Your snoring rate (45 - 100):", value=None, placeholder="Type a number...")
st.write('The current number is ', number)


