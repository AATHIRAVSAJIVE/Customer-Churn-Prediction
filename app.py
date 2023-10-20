from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model and scaler
with open('trained_logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the one-hot encoded DataFrame
with open('df_clean_encoded.pkl', 'rb') as df_file:
    df_clean = pickle.load(df_file)

# Define categorical and numerical columns
categorical_columns = ['Gender_Female', 'Gender_Male', 'Location_Chicago', 'Location_Houston', 'Location_Los Angeles', 'Location_Miami', 'Location_New York']
numerical_columns = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']

# Define a route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    user_input = {
        'Age': float(request.form['Age']),
        'Subscription_Length_Months': float(request.form['Subscription_Length_Months']),
        'Monthly_Bill': float(request.form['Monthly_Bill']),
        'Total_Usage_GB': float(request.form['Total_Usage_GB']),
        'Gender_Female': int(request.form['Gender'] == 'Female'),
        'Gender_Male': int(request.form['Gender'] == 'Male'),
        'Location_Chicago': int(request.form['Location'] == 'Chicago'),
        'Location_Houston': int(request.form['Location'] == 'Houston'),
        'Location_Los Angeles': int(request.form['Location'] == 'Los Angeles'),
        'Location_Miami': int(request.form['Location'] == 'Miami'),
        'Location_New York': int(request.form['Location'] == 'New York')
    }

    # Create a DataFrame from the user input
    input_data = pd.DataFrame([user_input])

    # Apply the same preprocessing steps as during training
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

   # Make predictions
    prediction = model.predict(input_data)[0]

     # Map prediction to text
    prediction_text = "Churned" if prediction == 1 else "Not Churned"

    return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True, port=5002)