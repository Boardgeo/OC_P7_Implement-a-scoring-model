import streamlit as st
import shap
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from urllib.request import urlopen
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from PIL import Image 
import time

# Streamlit settings
st.set_page_config(page_title="Loan Prediction App", layout = "wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the trained model and test dataset as database
model = pickle.load(open("models/Tuned_LGBM_50N.p", "rb"))
df_text = pickle.load(open("models/Test_clean_10k.p", "rb"))

# Get client data from the API
#url_data = "https://loan-predict-file.herokuapp.com/data"
#json_url_all = urlopen(url_data)
#data = json.loads(json_url_all.read())
#data_df = pd.DataFrame(data)

# get list of clients
# client_list = data_df['SK_ID_CURR'].tolist()

X = df_text.drop(['SK_ID_CURR'],axis=1)
threshold = 0.424
y_proba = model.predict_proba(X)[:,1]
y_predict = y_proba >= threshold
y_predict = np.array(y_predict > 0)*1
df_pred = df_text.copy()
df_pred['proba'] = y_proba
df_pred['prediction'] = y_predict


# get list of clients ID
client_list = df_pred['SK_ID_CURR'].tolist()

#logo = Image.open(("models/logo_pret.png"))

# Set home page
# Page title

st.columns(3)[1].title("Loan Prediction App")
st.columns(3)[1].header("Get applicant's score")


st.subheader("Applicant's Identification number")
client_id = st.selectbox("Six digit numbers", (client_list))


# Get client's data from the ID
client_data = df_pred[df_pred.SK_ID_CURR==int(client_id)]
col1, col2 = st.columns(2)
with col1:
    st.write('__Personal Details__')
    st.write('Gender:', client_data['CODE_GENDER'].values[0])
    st.write('Age:', round(client_data['DAYS_BIRTH'].values[0]))
    st.write('Education Level:', client_data['NAME_EDUCATION_TYPE'].values[0])
    st.write('Occupation:', client_data['OCCUPATION_TYPE'].values[0])
    st.write('Marital Status:', client_data['NAME_FAMILY_STATUS'].values[0])
    st.write('Number of Children:', client_data['CNT_CHILDREN'].values[0])
    st.write('Employment Duration (months):', round(abs(client_data['DAYS_EMPLOYED'].values[0]/30)))
    
with col2:
    st.write('__Credit Information__')
    st.write('Monthly Income:', client_data['AMT_INCOME_TOTAL'].values[0])
    st.write('Loan Amount:', client_data['AMT_CREDIT'].values[0])
    st.write('Annual Loan:', client_data['AMT_ANNUITY'].values[0])
    st.write('Credit Income Ratio:', round(client_data['CREDIT_INCOME_RATIO'].values[0], 1))
    st.write('Credit Repayment Rate:', round(client_data['PAYMENT_RATE'].values[0], 1))
    st.write('External_1:', round(client_data['EXT_SOURCE_1'].values[0], 2))
    st.write('External_2:', round(client_data['EXT_SOURCE_2'].values[0], 2))
    

#Retrieve the prediction and probability scores from the API
#url_api = "https://loan-predict-file.herokuapp.com/
#url_predict =  url_api + "predict/" + str(client_id)
#json_url_pred = urlopen(url_predict)
#result = json.loads(json_url_pred.read())
#Prediction = result["Prediction"] 
#Score = result["Score"]
#dict_ID = result[client_id]
#Predict_ID = pd.DataFrame.from_dict(eval(dict_ID))

decision_df = df_pred[df_pred["SK_ID_CURR"] == int(client_id)][["proba", "prediction"]]
score = round(decision_df['proba'].iloc[0]*100, 2)

predict_button = st.button("Predict")
if predict_button:

    # Construct the scoring gauge
    st.spinner('Scoring gauge loading.......')
    fig = go.Figure()
    fig.add_trace(go.Indicator(
            domain = {'x': [0,1], 'y': [0,1]},
            # client's score in %
            value = score,
            mode = "gauge+number",
            delta = {'reference': 50},
            gauge = {'axis':{'range': [None, 100], 'tickwidth': 3, 'tickcolor': 'darkblue'},
                'bar': {'color': 'white', 'thickness': 0.15},
                'bgcolor': 'white', 'borderwidth': 2, 'bordercolor': 'gray',
                'steps': [{'range': [0, 38], 'color': 'green'},
                            {'range': [38, 42.5], 'color': 'limeGreen'},
                            #{'range': [42.5, 43], 'color': 'red'},
                            {'range': [42.5, 50], 'color': 'orange'},
                            {'range': [50, 100], 'color': 'crimson'}],
                'threshold': {'line': {'color': 'red', 'width': 5}, 'thickness': 1.0, 'value': 42.4 }}))
    fig.update_layout(paper_bgcolor='white',
                                height=300, width=300,
                                font={'color': 'darkblue', 'family': 'Arial'},
                                margin=dict(l=0, r=0, b=0, t=0, pad=0))
    st.plotly_chart(fig, use_container_width=True)
    st.columns(3)[1].subheader("Customer's Scoring Gauge")

    if score <= 40:
        st.markdown("The applicant has a good potential to repay the loan")
        st.success('Decision: :green[Application Granted]')   

    elif 40 < score <=45:
        st.markdown("Risk of default! The applicant may not repay the loan")
        st.success('Decision: :orange[Application may be granted after providing additional information]') 

    elif 45 < score < 50:
        st.markdown("The applicant may likely not repay the loan")
        st.success('Decision: :orange[Ask for additional infomation or seek approval from your manager]') 
        
    else: 
        st.markdown(":red[Alert!] The applicant has a high risk to not repay the loan!")
        st.error('Decision: :red[Application Refused]')


#******************************************* GLOBAL INTERPRETATIONS****************************************************
# upload explainer, shap_values, and encoded dataframe for shap Interpretation
#explainer_G = pickle.load(open("models/explainer_G.p", "rb"))

# import shap values of the transformed data
shap_values = pickle.load(open("models/shap_values_L.p", "rb"))
columns_list = pickle.load(open("models/all_columns_50N.p","rb"))

# Transform input test data for shap interpretation
#X_transform = model['preprocessor'].transform(X)
#X_trans_df = pd.DataFrame(data = X_transform, columns = columns_list)
#explainer = shap.TreeExplainer(model['classifier'], X_trans_df)
#shap_values = explainer(X_trans_df, check_additivity=False)

option = st.sidebar.selectbox("Important Features", ["Individual", "Global", "Similar Profiles"])

# Summary plot of feature components according to their importance
st.columns(3)[1].header('Global Interpretability')
fig = plt.subplots(figsize=(6,4))
fig = shap.plots.beeswarm(shap_values, max_display=15)
st.pyplot(fig)

# Extract full information of client(s)
def feature_components():
    st.write('Sample size:')
    nb_client = st.number_input(label = 'Number of clients', min_value = 1, 
    max_value = df_text.shape[0], format = '%i')

    if st.button('Get sample'):
        st.write(df_text.sample(nb_client))

feature_components()

#******************************************* Distribution of features *******************************************************
                                                # Numerical features

st.subheader("Distribution of Quantitative Features")
df = X.copy()

# extract column lists of numerical features
num_cols =list(df.select_dtypes(include = ['float', 'int']).columns)
num_input = st.selectbox("Select a feature for interactive analysis. The client's position is represented by the red marker:", num_cols)
st.markdown(num_input)

x0 = df_pred[df_pred['prediction'] == 0][num_input]
y0 = df_pred[df_pred['prediction'] == 1][num_input]
z0 = df_pred[num_input]
bins = np.linspace(0, 1, 15)

num_client = df_pred[df_pred["SK_ID_CURR"] == (client_id)][num_input].item()

group_labels = ['Credit worthy', 'Non-credit worthy','Global']

fig = go.Figure()
fig.add_trace(go.Histogram(x=x0, name = 'Non defaulters'))
fig.add_trace(go.Histogram(x=y0, name = 'Defaulters'))
fig.add_trace(go.Histogram(x=z0,name = 'All the clients' ))
fig.add_vline(x= num_client, annotation_text = 'client n° '+ str(client_id), line_color = "red")
fig.update_layout(barmode='relative')
fig.update_traces(opacity=0.75)
plt.show()
st.plotly_chart(fig, use_container_width=True)


#*********************************************** Categorical features**********************************
st.subheader("Distribution of Quantitative Features")
# extract column lists of numerical features
cat_cols =list(df.select_dtypes(include = ['object']).columns)
cat_input = st.selectbox("Select a feature for interactive analysis. The client's position is represented by the red marker:", cat_cols)
st.subheader(cat_input)

sizes0 = list(df_pred[cat_input][df_pred['prediction']==0].value_counts().values)
labels0 =list(df_pred[cat_input][df_pred['prediction']==0].value_counts().index)

sizes1 = list(df_pred[cat_input][df_pred['prediction']==1].value_counts().values)
labels1 =list(df_pred[cat_input][df_pred['prediction']==1].value_counts().index)

size = list(df_pred[cat_input].value_counts().values)
labels = list(df_pred[cat_input].value_counts().index)


risque_client=df_pred[df_pred['SK_ID_CURR']== client_id][cat_input].item()

fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels0, values = sizes0, name = 'Non-defaulters' ), 1,1)
fig.add_trace(go.Pie(labels=labels1, values = sizes1, name ='Defaulters' ),1,2)
fig.add_trace(go.Pie(labels=labels, values = size, name = 'All the clients'),1,3)

fig.update_traces(hole=.45, hoverinfo="label+percent+name")

fig.update_layout(
    title_text="Distribution of clients",
    # Add annotations in the center of the pie chart.
    annotations=[dict(text='Non-defaulters', x=0.1, y=0.5, font_size=10, showarrow=False),
                 dict(text='Defaulters', x=0.5, y=0.5, font_size=10, showarrow=False),
                dict(text='Global', x=0.89, y=0.5, font_size=10, showarrow=False)])
st.plotly_chart(fig, use_container_width=False, sharing="streamlit",)

#******************************************* LOCAL/INDIVIDUAL INTERPRETATIONS******************************************************

explainer_L = pickle.load(open("models/explainer_G.p", "rb"))
Text_encode = pickle.load(open("models/Encoded_Text.p", "rb"))

# close neighbours

client_detail = df_pred[df_pred.SK_ID_CURR== int(client_id)]

# Client
no_child = client_detail['CNT_CHILDREN']
gender = client_detail['CODE_GENDER']
age = client_detail['DAYS_BIRTH'].item()
region = client_detail['REGION_RATING_CLIENT']
occupation = client_detail['OCCUPATION_TYPE']

# Close Neighbours
no_child_n = client_detail[client_detail['CNT_CHILDREN'] == no_child]
gender_n = client_detail[client_detail['CODE_GENDER'] == gender]
age_n = client_detail[client_detail['DAYS_BIRTH'] == age]
region_n = client_detail[client_detail['REGION_RATING_CLIENT'] == region]
occupation = client_detail[client_detail['OCCUPATION_TYPE'] == occupation]

if len(region_n) < 15:
    text_values = region_n.sample(len(region_n), random_state = 42)

if len(region_n >= 15):
    text_values = region_n.sample(15, random_state = 42)

fig, ax = plt.subplots(figsize = (10, 6))

plt.barh(range(len(text_values)),text_values['Proba'])
default_client=client_detail['Proba'].item()
plt.axhline(y=default_client,linewidth=8, color='#d62728')
plt.xlabel('% of default')
plt.ylabel('Num of similar profiles')
plt.figtext(0.755,0.855,'-',fontsize = 60,fontweight = 'bold',color = '#d62728')
plt.figtext(0.797,0.9,'Client '+str(client_id))
st.pyplot(fig)

n_mean=text_values['Proba'].mean()
diff_proba=round(abs(default_client-n_mean)*100,2)
st.write('The client',str(client_id),'has a difference of',str(diff_proba),'% of risk with clients of similar profiles.')



input_ID = Text_encode[Text_encode.SK_ID_CURR==int(client_id)]

idx = Text_encode.loc[Text_encode['SK_ID_CURR'] == int(client_id)].index[0]
data_idex = Text_encode.iloc[idx]
shap.plots.bar(shap_values[data_idex])


if st.checkbox("See the influencing factors"):
    with st.spinner("....loading the force plot"):
        input_shap = Text_encode.iloc[input_ID,:]
        X_shap_fp = input_shap.values.reshape(1,-1)
        shap_values_fp = explainer_L.shap_values(X_shap_fp)

        st.pyplot(shap.plots.force(explainer_L.expected_value[1], shap_values_fp[1], input_shap, matplotlib = True))
        st.pyplot()

        #st.decision_plot(explainer_L.expected_value[1], shap_values_fp[1,], )
#shap_values_L = pickle.load(open("models/shap_values_G.p", "rb"))
#X_encode_df = Text_encode.drop('SK_ID_CURR', axis = 1)
#shap_values_L = explainer(X_encode_df, check_additivity=False)

# use ID to get the position of the client in a corrensponding order on the dataframe
#idx = (Text_encode.loc[Text_encode["SK_ID_CURR"]==int(client_id)].index)

# display barplot of individual client
#x = Text_encode.iloc[client_id, :]
#x_array = x.values.reshape(1,-1)

#shap_values_x = explainer.shap_values(x_array)

#st.pyplot(shap.plots.force(explainer.expected_value[1], shap_values_x[1], x))
#shap.plots.bar(shap_values_L[idx])


#if st.checkbox("See infleuncing factors"):
   # with st.spinner("Display the important factors of the current client"):
        # set client's index
        #X_ind = shap_local.iloc[client_id,:]
        #X_ind_np = X_ind.values.reshape(1, -1)

        #shap_values_ind = explainer.shap


#df_pred

#X0 

# st.subheader(columns_list)

# Local Interpretation

