
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

new_data = pd.DataFrame([[]] ,columns=X.columns)

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


sr = st.number_input("Your snoring rate (45 - 100):", value=None, placeholder="Type a number...")


rr = st.number_input("Your respiration rate (16 - 30):", value=None, placeholder="Type a number...")


bt = st.number_input("Your body temperature rate (85 - 99 F):", value=None, placeholder="Type a number...")


lm = st.number_input("Your limb movement (4 - 19):", value=None, placeholder="Type a number...")


bo = st.number_input("Your blood oxygen (82 - 99):", value=None, placeholder="Type a number...")

em = st.slider("Eye movement", 60,105,65)

sh = st.slider("Sleeping hours", 0,12,8)

hr = st.slider("Heart rate", 50,85,65)

def fresh():
    new_data.add(sr)
    new_data.add(rr)
    new_data.add(bt)
    new_data.add(lm)
    new_data.add(bo)
    new_data.add(em)
    new_data.add(sh)
    new_data.add(hr)

if st.button('Check stress level'):
    st.write("workin on it... :)")
    fresh()
   
# Predict the stress level for the new data
    predicted_stress_level = decision_tree_cl.predict(new_data)

# Dictionary to map integer stress levels to human-readable labels
    stress_level_labels = {
        0: "Low/Normal",
        1: "Medium Low",
        2: "Medium",
        3: "Medium High",
        4: "High"
    }

# Assuming you already have the 'predicted_stress_level' from the previous code snippet
    predicted_stress_label = stress_level_labels[predicted_stress_level[0]]

# Display the human-readable label for the predicted stress level
    st.write("Predicted Stress Label for New Data:",predicted_stress_level[0],"(",predicted_stress_label,")")
