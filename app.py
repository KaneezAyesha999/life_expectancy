
from typing import Text
from flask import Flask, render_template, request
#import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
model = pickle.load(open('xg.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
    Country = request.form.get('Country')
    Year = request.form.get('Year')
    Status = request.form.get('Status')
    Adult_Mortality = request.form.get('Adult_Mortality')
    infant_deaths = request.form.get('infant_deaths')
    Alcohol = request.form.get('Alcohol')
    percentage_expenditure = request.form.get('percentage_expenditure')
    Hepatiits_B = request.form.get('Hepatiits_B')
    Measles = request.form.get('Measles')
    BMI = request.form.get('BMI')
    under_five_deaths = request.form.get('under_five_deaths')
    Polio = request.form.get('Polio')
    Total_expenditure = request.form.get('Total_expenditure')
    Diphtheria = request.form.get('Diphtheria')
    HIV_AIDS = request.form.get('HIV_AIDS')
    GDP = request.form.get('GDP')
    Population = request.form.get('Population')
    thinness_1_to_19_years = request.form.get('thinness_1_to_19_years')
    thinness_5_to_9_years = request.form.get('thinness_5_to_9_years')
    Income_composition_of_resources = request.form.get('Income_composition_of_resources')
    Schooling = request.form.get('Schooling')
    df = pd.DataFrame([[Country, Year, Status, Adult_Mortality, infant_deaths, Alcohol, percentage_expenditure, Hepatiits_B, Measles, BMI, under_five_deaths, Polio,
                        Total_expenditure, Diphtheria, HIV_AIDS, GDP, Population, thinness_1_to_19_years, thinness_5_to_9_years, Income_composition_of_resources, 
                        Schooling]])
    data = df
    categ = list(data.select_dtypes(include=['object']).columns)
    le = preprocessing.LabelEncoder()
    data[categ] = data[categ].apply(le.fit_transform)
    bools = list(data.select_dtypes(include=['bool']).columns)
    data[bools] = data[bools].astype(int)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    output = model.predict(df)
    return render_template('home.html', prediction_text='Life Expectancy {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
