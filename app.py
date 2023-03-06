from flask import Flask, jsonify, request, render_template
import pickle
import pandas as pd

#instantiate the Flask object
app = Flask(__name__)

# Load the model and test dataset
#model = pickle.load(open("models/Tuned_LGBM_100k.p", "rb"))
#Test_df = pickle.load(open("models/X_10k_backup.p", "rb"))

model = pickle.load(open("models/Tuned_LGBM_50N.p", "rb"))
Test_df = pickle.load(open("models/Test_clean_10k.p", "rb"))


# home page
@app.route('/')
def welcome():
    return "Enter a client ID in the URL bar"

# Create API routing call to get clients data and prediction
@app.route('/predict/<client_id>', methods = ['GET'])
def predict(client_id):
    result = {}
    client_id = int(client_id)
    
    if client_id not in list(Test_df['SK_ID_CURR']):
        result['message'] = 'Error! Please try again. This ID is not in the database'
        
        
    else:
        search_ID = Test_df[Test_df['SK_ID_CURR']==int(client_id)]
        
        # get prediction probablilities
        y_proba = model.predict_proba(search_ID.drop('SK_ID_CURR',axis=1))[0][1]
        
        # Define the optimal threshold - default (1) if greater than threshold, else non-default(0)
        threshold = pickle.load(open("models/Thres_opt.p", "rb"))

        # get the predictions
        if y_proba >= threshold:
            prediction = "Loan not granted, risk of default"

        else:
            prediction = "Loan granted"
      
        result["Score"] = round(y_proba, 2) 
        result["Prediction"] = prediction
        
    # Return JSON version of prediction
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
