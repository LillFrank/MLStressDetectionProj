
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


st.title("Human Stress Detection in and through Sleep")
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Snoring rate", "Respiration rate", "Body temperature", "Limb movement", "Blood oxygen", "Eye movement"])
with tab1:
   st.header("Snoring rate")
   st.write("The rate or intensity of snoring during sleep, which could be measured in some unit or scale.")

with tab2:
   st.header("Respiration rate")
   st.write("The number of breaths taken per minute during sleep.")

with tab3:
   st.header("Body temeprature")
   st.write("The body temperature of the user during sleep, possibly measured in degrees Celsius or Fahrenheit.")

with tab4:
   st.header("Limb movement")
   st.write("The rate or intensity of limb movement during sleep, indicating how active or restless the person is.")

with tab5:
   st.header("Blood oxygen")
   st.write("The blood oxygen level, which represents the amount of oxygen present in the blood during sleep.")

with tab6:
   st.header("Eye movement")
   st.write("The eye movement activity during sleep, which might indicate the Rapid Eye Movement (REM) phase of sleep.")


sr = st.number_input("Your snoring rate (45 - 100):", value=None, placeholder="Type a number...")
st.write('The current number is ', sr)

rr = st.number_input("Your respiration rate (45 - 100):", value=None, placeholder="Type a number...")
st.write('The current number is ', sr)

bt = st.number_input("Your body temperature rate (45 - 100):", value=None, placeholder="Type a number...")
st.write('The current number is ', sr)

lm = st.number_input("Your limb movement (45 - 100):", value=None, placeholder="Type a number...")
st.write('The current number is ', sr)

bo = st.number_input("Your blood oxygen (45 - 100):", value=None, placeholder="Type a number...")
st.write('The current number is ', sr)


