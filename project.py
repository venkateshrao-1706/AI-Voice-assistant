import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout,Dense,Input
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
scaler = StandardScaler()
model = Sequential()

df = pd.DataFrame(pd.read_csv("C:/Users/venka/OneDrive/Documents/smoking_driking_dataset_Ver01.csv"))

print(df.duplicated().sum())
print(df.isna().sum())
df = df.dropna()
df = pd.get_dummies(df,columns=['DRK_YN','sex'])

null_precentage = (df.isnull().sum()/len(df))*100
print(null_precentage.sort_values(ascending=False))
print(df.sample(20))

X = df.drop(columns=['DRK_YN_N', 'DRK_YN_Y'])
y = df[['DRK_YN_Y']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.15,random_state=45,stratify =y)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

callbacks = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    verbose=1,
    restore_best_weights = True
)


from tensorflow.keras.regularizers import l2
model = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(30, activation="relu",
          kernel_regularizer=l2(0.001)),

    Dense(20, activation="relu",
          kernel_regularizer=l2(0.001)),

    Dropout(0.25),

    Dense(29, activation="relu",
          kernel_regularizer=l2(0.001)),

    Dense(1, activation="sigmoid")
])


from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.005)
model.compile(loss='binary_crossentropy',
              optimizer = optimizer,
              metrics=['accuracy']

              )


history = model.fit(X_train_scaled,y_train,batch_size=400,epochs=200,validation_split=0.2 , callbacks=[callbacks])

model.save(
    "ANN_FOR_SOMKING_&_DRINKING_PREDICTION.keras"
)

# Save fitted scaler
import joblib
joblib.dump(
    scaler,
    "scaler.pkl"
)

y_prob = model.predict(X_test_scaled)

y_pred = (y_prob >= 0.5).astype(int).ravel()

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='Training accuracy')
plt.plot(history.history['val_accuracy'],label='val accuracy')
plt.legend()


import streamlit  as st
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Drinking Status Prediction",
    page_icon="🍷",
    layout="wide"
)


@st.cache_resource
def loader():

 model = load_model("ANN_FOR_SOMKING_&_DRINKING_PREDICTION.keras")

 scaler = joblib.load("scaler.pkl")

 return model,scaler

model,scaler = loader()

st.title("Drinking Status Prediction")

st.write("Enter the Health information Below")

with st.form('prediction form'):

 age = st.number_input("Age", 18, 100, 30)

 height = st.number_input(
    "Height (cm)",
    100.0,
    220.0,
    170.0
  )

 weight = st.number_input(
    "Weight (kg)",
    30.0,
    200.0,
    70.0
 )

 waistline = st.number_input(
    "Waistline",
    value=80.0
 )

 sight_left = st.number_input(
    "Sight Left",
    value=1.0
 )

 sight_right = st.number_input(
    "Sight Right",
    value=1.0
 )

 hear_left = st.number_input(
    "Hearing Left",
    value=1.0
)

 hear_right = st.number_input(
    "Hearing Right",
    value=1.0
 )

 SBP = st.number_input(
    "Systolic Blood Pressure",
    value=120.0
 )

 DBP = st.number_input(
    "Diastolic Blood Pressure",
    value=80.0
 )

 if SBP <= DBP:

        st.warning(
            "Systolic BP is normally expected to be higher than Diastolic BP."
        )

 BLDS = st.number_input(
    "Blood Sugar",
    value=100.0
 )

 tot_chole = st.number_input(
    "Total Cholesterol",
    value=180.0
 )

 HDL_chole = st.number_input(
    "HDL Cholesterol",
    value=50.0
 )

 LDL_chole = st.number_input(
    "LDL Cholesterol",
    value=100.0
 )

 triglyceride = st.number_input(
    "Triglyceride",
    value=120.0
 )

 hemoglobin = st.number_input(
    "Hemoglobin",
    value=14.0
 )

 urine_protein = st.number_input(
    "Urine Protein",
    value=1.0
 )

 serum_creatinine = st.number_input(
    "Serum Creatinine",
    value=1.0
 )

 SGOT_AST = st.number_input(
    "SGOT AST",
    value=25.0
 )

 SGOT_ALT = st.number_input(
    "SGOT ALT",
    value=25.0
 )

 gamma_GTP = st.number_input(
    "Gamma GTP",
    value=30.0
 )

 smoking = st.selectbox( "Smoking Status",
    [1.0, 2.0, 3.0])

 sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
 )
 submitted = st.form_submit_button('Predict')

sex_Female = 1  if sex =="Female" else 0
sex_Male =   1 if sex =="Male" else 0

# Prediction

if submitted:                                             

    input_data = pd.DataFrame([{

         "age": age,
        "height": height,
        "weight": weight,
        "waistline": waistline,
        "sight_left": sight_left,
        "sight_right": sight_right,
        "hear_left": hear_left,
        "hear_right": hear_right,
        "SBP": SBP,
        "DBP": DBP,
        "BLDS": BLDS,
        "tot_chole": tot_chole,
        "HDL_chole": HDL_chole,
        "LDL_chole": LDL_chole,
        "triglyceride": triglyceride,
        "hemoglobin": hemoglobin,
        "urine_protein": urine_protein,
        "serum_creatinine": serum_creatinine,
        "SGOT_AST": SGOT_AST,
        "SGOT_ALT": SGOT_ALT,
        "gamma_GTP": gamma_GTP,
        "SMK_stat_type_cd": smoking,
        "sex_Female": sex_Female ,
        "sex_Male": sex_Male
    }])

    with st.spinner("Analyzing health information..."):


     input_scaled = scaler.transform(input_data)

     probability = model.predict(input_scaled,verbose = 0)[0][0]

     prediction = 1 if probability >=0.5 else 0

     if prediction == 1 :
        st.error("Drinking habit Detected, Drinker")
     else:
        st.success("Drinking habit not Detected, Non Drinker")
    
    st.write("Drinking probablity:",
        round(float(probability)*100,2),"%")

    tab1,tab2 = st.tabs(['prediction','Input Summary'])
    with tab1:

     probability = float(probability)

     non_drinker_probability = (
        1 - probability)

     if probability >= 0.5:
      st.warning(
                "🍷 Predicted Status: Drinker"
            )

     else:
      st.success(
                "✅ Predicted Status: Non-Drinker"
            )
     chart_data = pd.DataFrame({
    'status':['Drinker','Non-Drinker'],
     'probablity':[probability*100,non_drinker_probability*100
    ]
 })
     st.bar_chart(chart_data,x="status",y="probablity")

     st.progress(probability)

    with tab2:

        st.subheader(
            "Entered Information"
        )

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )



