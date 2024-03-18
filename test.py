import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.metrics import accuracy_score
from joblib import dump,load
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings 
warnings.filterwarnings('ignore')
def wordopt(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove digits
    text = re.sub(r'\d', '', text)
    
    # Remove newline characters
    text = re.sub(r'\n', ' ', text)
    
    return text

vectorization = load('tfidf_vectorizer.pkl')
LR = joblib.load('logistic_regression.pkl')
DT = joblib.load('DT.pkl')
GBC = joblib.load('GBC.pkl')
RFC = joblib.load('RFC.pkl')
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    def predict_labels(news_df):
        news_df["text"].fillna("", inplace=True)
        
        # Apply preprocessing steps and vectorization
        news_df["text"] = news_df["text"].apply(wordopt)
        new_x_test = news_df["text"]
        new_xv_test = vectorization.transform(new_x_test)
    
        # Predict using each model
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GBC = GBC.predict(new_xv_test)
        pred_RFC = RFC.predict(new_xv_test)
    
        # Create a DataFrame to hold the predictions
        predictions_df = pd.DataFrame({
            "Logistic Regression": [output_lable(pred_LR[i]) for i in range(len(pred_LR))],
            "Decision Tree": [output_lable(pred_DT[i]) for i in range(len(pred_DT))],
            "Gradient Boosting": [output_lable(pred_GBC[i]) for i in range(len(pred_GBC))],
            "Random Forest": [output_lable(pred_RFC[i]) for i in range(len(pred_RFC))]
        })
    
        return predictions_df

x_test = pd.read_csv("Data/x_test.csv")

# Perform predictions on the test data
predictions = predict_labels(x_test)
print(predictions)
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),output_lable(pred_DT[0]), 
                                                                                                              
                                                                                                        output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))

news = str(input())
manual_testing(news)
    
    
