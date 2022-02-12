import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import random
import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split ,cross_val_score , KFold
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
import streamlit as st
from bokeh.models.widgets import Div
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

model_results = st.sidebar.checkbox('SHOW MODEL RESULTS')

algorithms = ('','DecisionTreeClassifier' ,'LogisticRegression' ,'RandomForestClassifier' ,'CatBoostClassifier' ,'SVMCLassifier')
ml_algorithm = st.sidebar.selectbox("ML ALGORITHM" , algorithms)

st.write("""
# Football  Player  Ranking  Project
""")

st.text("")

image = Image.open("https://github.com/UGURSELIMOZEN/ISTDSA_DS_Bootcamp/blob/main/Project3_Classification_SQL/FootballPlayerRankingApp/mbappe.png")
st.image(image, use_column_width = True)

st.text("")

df = pd.read_csv("Players.csv")

abc = sns.histplot(data=df, x="overall_rating")

for i in range(df.shape[0]) :

    if (df.loc[i, "overall_rating"] >= 42.0) & (df.loc[i , "overall_rating"] < 85.0) :
        df.loc[i, "overall_rating"] = 0

    elif (df.loc[i, "overall_rating"] >= 85.0) :
        df.loc[i, "overall_rating"] = 1

df["overall_rating"] = df["overall_rating"].astype("int64")


df["attacking_work_rate"].fillna("medium", inplace = True)
col_list = ["volleys","curve","agility","balance","jumping","vision","sliding_tackle"]
for item in col_list :
    df[item].fillna(round(df[item].mean() , 0) , inplace = True)

selected_columns = ["potential" , "reactions" , "short_passing" , "vision" , "long_passing" , "ball_control" , "shot_power" , "long_shots" , "curve" , "dribbling" , "crossing" , "volleys" , "positioning" , "free_kick_accuracy" , "penalties" ,
"aggression" , "finishing" , "stamina" , "heading_accuracy"  , "overall_rating"]

modelling_df = df[selected_columns]

trainData = modelling_df.tail(8678)
testData =  modelling_df.head(2170)

X = trainData.drop("overall_rating", 1)
y = trainData.overall_rating

X_test = testData.drop("overall_rating", 1)
y_test = testData.overall_rating

sc = StandardScaler()

X = pd.DataFrame(sc.fit_transform(X))
X_test = pd.DataFrame(sc.transform(X_test))

oversample = SMOTE( random_state=42)
X_train, y_train = oversample.fit_resample(X, y)

chck = pd.DataFrame()
chck['overall_rating'] = y_train

def DTClassifier(X,y,X1,y1,valueList) :
    st.text("")
    st.text("")
    st.subheader("Decision Tree Classifier Model Results")
    dtc=DecisionTreeClassifier(random_state=42)
    dtc.fit(X, y)
    preds = dtc.predict(X1)
    score = dtc.score(X1, y1)
    st.text('Model Report:\n ' + classification_report(y1, preds))
    cm = confusion_matrix(y1, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True)
    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y1, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.subheader("Decision Tree Classifier Prediction Result")
    prediction = dtc.predict([valueList])
    if prediction == 1 :
        st.write("A class football player !")
    else :
        st.write("B class football player !")


def LRClassifier(X,y,X1,y1,valueList) :
    st.text("")
    st.text("")
    st.subheader("Logistic Regression Model Results")
    lgr=LogisticRegression(random_state=42 , max_iter = 200)
    lgr.fit(X, y)
    preds = lgr.predict(X1)
    score = lgr.score(X1, y1)
    st.text('Model Report:\n ' + classification_report(y1, preds))
    cm = confusion_matrix(y1, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True)
    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y1, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    st.pyplot()
    st.subheader("Logistic Regression Prediction Result")
    prediction = lgr.predict([valueList])
    if prediction == 1 :
        st.write("A class football player !")
    else :
        st.write("B class football player !")


def RFClassifier(X,y,X1,y1,valueList) :
    st.text("")
    st.text("")
    st.subheader("Random Forest Classifier Model Results")
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X, y)
    preds = rfc.predict(X1)
    score = rfc.score(X1, y1)
    st.text('Model Report:\n ' + classification_report(y1, preds))
    cm = confusion_matrix(y1, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True)
    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y1, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    st.pyplot()
    st.subheader("Random Forest Classifier Prediction Result")
    prediction = rfc.predict([valueList])
    if prediction == 1 :
        st.write("A class football player !")
    else :
        st.write("B class football player !")


def CBClassifier(X,y,X1,y1,valueList) :
    st.text("")
    st.text("")
    st.subheader("CatBoost Classifier Model Results")
    cbc=CatBoostClassifier(n_estimators = 200, max_depth = 5, verbose = 0 , random_state=42)
    cbc.fit(X, y)
    preds = cbc.predict(X1)
    score = cbc.score(X1, y1)
    st.text('Model Report:\n ' + classification_report(y1, preds))
    cm = confusion_matrix(y1, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True)
    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y1, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    st.pyplot()
    st.subheader("CatBoost Classifier Prediction Result")
    prediction = cbc.predict([valueList])
    if prediction == 1 :
        st.write("A class football player !")
    else :
        st.write("B class football player !")


def SVMClassifier(X,y,X1,y1,valueList) :
    st.text("")
    st.text("")
    st.subheader("SVM Classifier Model Results")
    svc = SVC(random_state=42)
    svc.fit(X, y)
    preds = svc.predict(X1)
    score = svc.score(X1, y1)
    st.text('Model Report:\n ' + classification_report(y1, preds))
    cm = confusion_matrix(y1, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True)
    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y1, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    st.pyplot()
    st.subheader("SVM CLassifier Prediction Result")
    prediction = svc.predict([valueList])
    if prediction == 1 :
        st.write("A class football player !")
    else :
        st.write("B class football player !")



if (model_results == False) :

    st.subheader("About Data")
    st.text("")

    st.write("""
    ##### The Ultimate Soccer database for data analysis and machine learning

    What you get:

    •  +25,000 matches

    •  +10,000 players

    •  11 European Countries with their lead championship

    •  Seasons 2008 to 2016

    •  Players and Teams' attributes* sourced from EA Sports' FIFA video game series, including the weekly updates

    •  Team line up with squad formation (X, Y coordinates)

    •  Betting odds from up to 10 providers

    •  Detailed match events (goal types, possession, corner, cross, fouls, cards etc…) for +10,000 matches

    """)

    st.text("")

    if st.button('👉🏻 Data Source 👈🏻'):
        js = "window.open('https://www.kaggle.com/hugomathien/soccer')"  # New tab or window
        js = "window.location.href = 'https://www.kaggle.com/hugomathien/soccer'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)



    st.text("")
    st.text("")
    st.text("")
    st.subheader("Overall Rating Distribution")
    abc
    st.pyplot()

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")


    st.subheader("Overall Rating Correlation Heatmap")
    k = 10 #number of variables for heatmap
    corrmat = df.corr()
    cols = corrmat.nlargest(k, 'overall_rating')['overall_rating'].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    plt.figure(figsize=(20,20))
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 13}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    st.pyplot()


    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.subheader("Imbalanced Data Distribution")
    plt.figure(figsize=(10,8))
    sns.countplot(modelling_df['overall_rating'])
    plt.show()
    st.pyplot()

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.subheader("Oversampled Data Distribution")
    plt.figure(figsize=(10,8))
    sns.countplot(chck['overall_rating'])
    plt.show()
    st.pyplot()


elif model_results == True :

    potential = st.sidebar.slider(' potential ', 0, 100)
    reactions = st.sidebar.slider(' reactions ', 0, 100)
    vision = st.sidebar.slider(' vision ', 0, 100)
    volleys = st.sidebar.slider(' volleys ', 0, 100)
    penalties = st.sidebar.slider(' penalties ', 0, 100)
    long_passing = st.sidebar.slider(' long_passing ', 0, 100)
    short_passing = st.sidebar.slider(' short_passing ', 0, 100)
    ball_control = 50
    curve = 50
    finishing =  50
    free_kick_accuracy = 50
    positioning =  50
    long_shots = 50
    dribbling = 50
    crossing = 50
    shot_power = 50
    aggression = 50
    stamina = 50
    heading_accuracy = 50

    new = [potential,reactions,vision,volleys,penalties,long_passing,short_passing,ball_control,curve,finishing,free_kick_accuracy,positioning,long_shots,dribbling,crossing,shot_power,aggression,stamina,heading_accuracy ]



    if  (ml_algorithm == 'DecisionTreeClassifier') :
        DTClassifier(X_train,y_train,X_test,y_test,new)

    elif (ml_algorithm == 'LogisticRegression') :
        LRClassifier(X_train,y_train,X_test,y_test,new)

    elif (ml_algorithm == 'RandomForestClassifier') :
        RFClassifier(X_train,y_train,X_test,y_test,new)

    elif (ml_algorithm == 'CatBoostClassifier') :
        CBClassifier(X_train,y_train,X_test,y_test,new)

    elif (ml_algorithm == 'SVMCLassifier') :
        SVMClassifier(X_train,y_train,X_test,y_test,new)
