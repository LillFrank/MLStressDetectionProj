
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
st.header(divider='rainbow')


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



sr = st.number_input("Your snoring rate (45 - 100):",min_value=45, value="min", placeholder="Type a number...")


rr = st.number_input("Your respiration rate (16 - 30):",min_value=16, value="min", placeholder="Type a number...")


bt = st.number_input("Your body temperature rate (85 - 99 F):",min_value=85, value="min", placeholder="Type a number...")


lm = st.number_input("Your limb movement (4 - 19):", value="min",min_value=4, placeholder="Type a number...")


bo = st.number_input("Your blood oxygen (82 - 99):", value="min",min_value=82, placeholder="Type a number...")


em = st.number_input("Eye movement (60 - 105):", value="min",min_value=60, placeholder="Type a number...")



sh = st.number_input("Sleeping hour (0 - 12):", value="min",min_value=0, placeholder="Type a number...")



hr = st.number_input("Heart rate (50 - 85):", value="min",min_value=50, placeholder="Type a number...")

 
def reload():
   new_data = pd.DataFrame([[sr,rr,bt,lm,bo,em,sh,hr]], columns=X.columns)
   return new_data



   

if st.button('Check stress level'):
   
   
   data = reload()
   predicted_stress_level = decision_tree_cl.predict(data)


   stress_level_labels = {
        0: ":green[Low/Normal] :sunglasses: ",
        1: ":blue[Medium Low] :no_mouth: ",
        2: ":orange[Medium] :neutral_face:",
        3: ":violet[Medium High] :anguished:",
        4: ":red[High] :worried:"
   }


   predicted_stress_label = stress_level_labels[predicted_stress_level[0]]


   st.subheader("Your Predicted Stress Level:")
   st.write( predicted_stress_level[0],", ",predicted_stress_label,)
    
