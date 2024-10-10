from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.feature_selection as fs
import sklearn.preprocessing as pp
import os

app = Flask(__name__)

# Load CSV data
data = pd.read_csv('credit_cards.csv')

@app.route('/')
def index():
    return render_template('index.html')

# Task 1: Check for missing values
@app.route('/missing-values', methods=['GET'])
def missing_values():
    missing_vals = pd.isnull(data).sum().to_dict()
    return jsonify(missing_vals)

# Task 1b: Plot distribution of the class attribute
@app.route('/default-distribution', methods=['GET'])
def default_distribution():
    height = data['DEFAULT'].value_counts()
    x_axis = np.arange(2)
    plt.bar(x_axis, height)
    plt.text(0, height[0]+100, str(height[0]))
    plt.text(1, height[1]+100, str(height[1]))
    plt.title('Default Distribution in Credit Card Clients')
    plt.savefig('static/default_distribution.png')
    plt.close()
    return render_template('chart.html', image='default_distribution.png')

# Task 2c: Chi-Square analysis and ranking
@app.route('/chi-square-ranking', methods=['GET'])
def chi_square_ranking():
    chi2 = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    feat_chi2 = pd.DataFrame(data, columns=chi2)
    pays = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    for p in pays:
        feat_chi2[p] += 2
    chi2_score = fs.chi2(feat_chi2, data['DEFAULT'])
    chi_rank = [['SEX', chi2_score[0][0]], ['EDUCATION', chi2_score[0][1]], ['MARRIAGE', chi2_score[0][2]], 
                ['PAY_1', chi2_score[0][3]], ['PAY_2', chi2_score[0][4]], ['PAY_3', chi2_score[0][5]], 
                ['PAY_4', chi2_score[0][6]], ['PAY_5', chi2_score[0][7]], ['PAY_6', chi2_score[0][8]]]
    chi_rank.sort(key=lambda x: x[1], reverse=True)
    return jsonify(chi_rank)

# Task 2c: Mutual Information ranking
@app.route('/mutual-information-ranking', methods=['GET'])
def mutual_information_ranking():
    com = ['ID', 'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
           'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    feature_com = pd.DataFrame(data, columns=com)
    com_score = fs.mutual_info_classif(feature_com, data['DEFAULT'], random_state=156)
    com_rank = [['ID', com_score[0]], ['LIMIT_BAL', com_score[1]], ['AGE', com_score[2]], 
                ['BILL_AMT1', com_score[3]], ['BILL_AMT2', com_score[4]], ['BILL_AMT3', com_score[5]], 
                ['BILL_AMT4', com_score[6]], ['BILL_AMT5', com_score[7]], ['BILL_AMT6', com_score[8]], 
                ['PAY_AMT1', com_score[9]], ['PAY_AMT2', com_score[10]], ['PAY_AMT3', com_score[11]], 
                ['PAY_AMT4', com_score[12]], ['PAY_AMT5', com_score[13]], ['PAY_AMT6', com_score[14]]]
    com_rank.sort(key=lambda x: x[1], reverse=True)
    return jsonify(com_rank)

# Task 4: Normalize numerical features
@app.route('/normalize', methods=['GET'])
def normalize_features():
    numericals = ['ID', 'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                  'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'AVG_NET']
    numericals_data = pd.DataFrame(data, columns=numericals)
    scaler = pp.MinMaxScaler()
    norm = scaler.fit_transform(numericals_data)
    norm_data = pd.DataFrame(norm, columns=numericals)
    return norm_data.to_html()

if __name__ == '__main__':
    app.run(debug=True)
