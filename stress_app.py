
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Snoring rate", "Respiration rate", "Body temperature", "Limb movement", "Blood oxygen", "Eye movement", "Sleeping hours", "Heart rate"])
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

with tab7:
   st.header("Sleeping hours")
   st.write("The number of hours slept during a particular sleep session.")

with tab8:
   st.header("Heart rate")
   st.write("The number of heartbeats per minute during sleep, an essential physiological parameter related to overall health and sleep quality.")

new_data = pd.DataFrame([[100.0, 29.0, 99.0, 35.0, 90.0, 95.0, 2.0, 80.0]] ,columns=X.columns)

sr = st.number_input("Your snoring rate (45 - 100):", value=None, placeholder="Type a number...")
new_data[0]= sr

rr = st.number_input("Your respiration rate (16 - 30):", value=None, placeholder="Type a number...")
new_data[1]= rr

bt = st.number_input("Your body temperature rate (85 - 99 F):", value=None, placeholder="Type a number...")
new_data[2]= bt

lm = st.number_input("Your limb movement (4 - 19):", value=None, placeholder="Type a number...")
new_data[3]= lm

bo = st.number_input("Your blood oxygen (82 - 99):", value=None, placeholder="Type a number...")
new_data[4]= bo

em = st.number_input("Eye movement (60 - 105):", value=None, placeholder="Type a number...")
new_data[5]= em


sh = st.number_input("Sleeping hour (0 - 12):", value=None, placeholder="Type a number...")
new_data[6]= sh


hr = st.number_input("Heart rate (50 - 85):", value=None, placeholder="Type a number...")
new_data[7]= hr
 

new_data = new_data.astype(x_train.dtypes)
   

if st.button('Check stress level'):
   
   

    predicted_stress_level = decision_tree_cl.predict(new_data)


    stress_level_labels = {
        0: "Low/Normal",
        1: "Medium Low",
        2: "Medium",
        3: "Medium High",
        4: "High"
    }


    predicted_stress_label = stress_level_labels[predicted_stress_level[0]]


    st.write("Predicted Stress Label for New Data:", predicted_stress_level[0])
             #predicted_stress_level[0],"(",predicted_stress_label,")")
    
