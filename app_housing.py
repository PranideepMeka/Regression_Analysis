from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load models
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet',
    'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM', 'LGBM',
    'XGBoost', 'KNN'
]
models = {name: pickle.load(open(f'{name}.pkl', 'rb')) for name in model_names}

# Load evaluation results
results_df = pd.read_csv('model_evaluation_results.csv')

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    input_data = {
        'Avg. Area Income': float(request.form['Avg. Area Income']),
        'Avg. Area House Age': float(request.form['Avg. Area House Age']),
        'Avg. Area Number of Rooms': float(request.form['Avg. Area Number of Rooms']),
        'Avg. Area Number of Bedrooms': float(request.form['Avg. Area Number of Bedrooms']),
        'Area Population': float(request.form['Area Population'])
    }
    input_df = pd.DataFrame([input_data])

    if model_name in models:
        model = models[model_name]
        prediction = model.predict(input_df)[0]
        return render_template('results.html', prediction=prediction, model_name=model_name)
    else:
        return jsonify({'error': 'Model not found'}), 400

@app.route('/results')
def results():
    return render_template('model.html', tables=[results_df.to_html(classes='data')], titles=results_df.columns.values)

@app.route('/powerbi')
def powerbi():
    # Replace with your actual values from Power BI
    access_token = "YOUR_EMBED_TOKEN"
    embed_url = "https://app.powerbi.com/reportEmbed?reportId=YOUR_REPORT_ID"
    report_id = "YOUR_REPORT_ID"
    token_type = "1"  # "1" for Embed token

    return render_template(
        'powerbi.html',
        access_token=access_token,
        embed_url=embed_url,
        report_id=report_id,
        token_type=token_type
    )

if __name__ == '__main__':
    app.run(debug=True)
