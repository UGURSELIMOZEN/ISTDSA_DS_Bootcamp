import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import string
import streamlit as st
from bokeh.models.widgets import Div
from PIL import Image
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_option('deprecation.showPyplotGlobalUse', False)

review = st.sidebar.checkbox('ADD REVIEW TO MOVIE')
model_results = st.sidebar.checkbox('SHOW MODEL RESULTS')

technique = ('','Count Vectorizer' ,'TF-IDF Vectorizer')
algorithms = ('','LogisticRegression-Unigram' ,'LogisticRegression-Bigram' ,'MultinomialNB' ,'BernoulliNB')
nlp_technique = st.sidebar.selectbox("NLP TECHNIQUE" , technique)
ml_algorithm = st.sidebar.selectbox("ML ALGORITHM" , algorithms)


st.write("""
# MOVIE SENTIMENT ANALYSIS APP
""")

st.text("")

image = Image.open('godfather.jpg')
st.image(image, use_column_width = True)

st.text("")

df = pd.read_csv("movieReviews.csv")

fig1 = px.histogram(df, x="rating", height= 550, width=900)

reviews_meta = pd.DataFrame()
reviews_meta['is_spoiler'] = df['is_spoiler']
reviews_meta['has_word_spoiler'] = df['review_text'].apply(lambda text: 1 if 'SPOILER' in text.upper() else 0)

pie1 = reviews_meta['is_spoiler'].value_counts().reset_index().sort_values(by='index').replace({True: "Spoiler", False: "Not-Spoiler"})
pie2 = reviews_meta[reviews_meta['has_word_spoiler'] == 1]['is_spoiler'].value_counts().reset_index().sort_values(by='index').replace({True: "Spoiler", False: "Not-Spoiler"})

fig2 = px.pie(pie1, values='is_spoiler', names='index', title='All Reviews Spoiler Distribution',height=600)
fig3 = px.pie(pie2, values='is_spoiler', names='index', title="Reviews Containing word 'Spoiler'",height=600)

df['sentiment'] = np.where(df['rating'] >= 8, 'positive', 'negative')
bar1 = df[["_id","sentiment"]].groupby("sentiment").count().rename(columns={"_id":"comment_Number"}).reset_index()
fig4 = px.bar(bar1, x="sentiment", y="comment_Number", color="comment_Number", height= 500, width = 800)



def LogReg1(data) :

    data = data.tail(2000)
    data["sentiment"] = data["sentiment"].replace({"positive":1,"negative":0})
    X = data.review_text
    y = data.sentiment

    X_train, X_test, y_train, y_test = X[:1600], X[1600:], y[:1600], y[1600:]
    cv1 = CountVectorizer(stop_words='english')
    X_train_cv1 = cv1.fit_transform(X_train)
    X_test_cv1  = cv1.transform(X_test)

    lr = LogisticRegression()
    lr.fit(X_train_cv1, y_train)
    y_pred_cv1 = lr.predict(X_test_cv1)
    st.text("")
    st.text("")
    st.text("")
    st.text('Model Report:\n ' + classification_report(y_test, y_pred_cv1))
    cm = confusion_matrix(y_test, y_pred_cv1)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'],
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':10}, cmap="YlGnBu")

    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_cv1)
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


def LogReg2(data) :

    data = data.tail(2000)
    data["sentiment"] = data["sentiment"].replace({"positive":1,"negative":0})
    X = data.review_text
    y = data.sentiment

    X_train, X_test, y_train, y_test = X[:1600], X[1600:], y[:1600], y[1600:]
    cv1 = CountVectorizer(ngram_range=(1,2), binary=True, stop_words='english')
    X_train_cv1 = cv1.fit_transform(X_train)
    X_test_cv1  = cv1.transform(X_test)

    lr = LogisticRegression()
    lr.fit(X_train_cv1, y_train)
    y_pred_cv1 = lr.predict(X_test_cv1)
    st.text("")
    st.text("")
    st.text("")
    st.text('Model Report:\n ' + classification_report(y_test, y_pred_cv1))
    cm = confusion_matrix(y_test, y_pred_cv1)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'],
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':10}, cmap="YlGnBu")

    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_cv1)
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


def NB1(data) :

    data = data.tail(2000)
    data["sentiment"] = data["sentiment"].replace({"positive":1,"negative":0})
    X = data.review_text
    y = data.sentiment

    X_train, X_test, y_train, y_test = X[:1600], X[1600:], y[:1600], y[1600:]
    cv1 = CountVectorizer(stop_words='english')
    X_train_cv1 = cv1.fit_transform(X_train)
    X_test_cv1  = cv1.transform(X_test)

    mnb = MultinomialNB()
    mnb.fit(X_train_cv1, y_train)
    y_pred_cv1 = mnb.predict(X_test_cv1)
    st.text("")
    st.text("")
    st.text("")
    st.text('Model Report:\n ' + classification_report(y_test, y_pred_cv1))
    cm = confusion_matrix(y_test, y_pred_cv1)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'],
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':10}, cmap="YlGnBu")

    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_cv1)
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


def NB2(data) :

    data = data.tail(2000)
    data["sentiment"] = data["sentiment"].replace({"positive":1,"negative":0})
    X = data.review_text
    y = data.sentiment

    X_train, X_test, y_train, y_test = X[:1600], X[1600:], y[:1600], y[1600:]
    cv1 = CountVectorizer(ngram_range=(1,2), binary=True, stop_words='english')
    X_train_cv1 = cv1.fit_transform(X_train)
    X_test_cv1  = cv1.transform(X_test)

    bnb = BernoulliNB()
    bnb.fit(X_train_cv1, y_train)
    y_pred_cv1 = bnb.predict(X_test_cv1)
    st.text("")
    st.text("")
    st.text("")
    st.text('Model Report:\n ' + classification_report(y_test, y_pred_cv1))
    cm = confusion_matrix(y_test, y_pred_cv1)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'],
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':10}, cmap="YlGnBu")

    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_cv1)
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


def LogReg1_tfidf(data) :

    data = data.tail(2000)
    data["sentiment"] = data["sentiment"].replace({"positive":1,"negative":0})
    X = data.review_text
    y = data.sentiment

    X_train, X_test, y_train, y_test = X[:1600], X[1600:], y[:1600], y[1600:]
    cv1 = TfidfVectorizer(stop_words='english')
    X_train_cv1 = cv1.fit_transform(X_train)
    X_test_cv1  = cv1.transform(X_test)

    lr = LogisticRegression()
    lr.fit(X_train_cv1, y_train)
    y_pred_cv1 = lr.predict(X_test_cv1)
    st.text("")
    st.text("")
    st.text("")
    st.text('Model Report:\n ' + classification_report(y_test, y_pred_cv1))
    cm = confusion_matrix(y_test, y_pred_cv1)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'],
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':10}, cmap="YlGnBu")

    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_cv1)
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


def LogReg2_tfidf(data) :

    data = data.tail(2000)
    data["sentiment"] = data["sentiment"].replace({"positive":1,"negative":0})
    X = data.review_text
    y = data.sentiment

    X_train, X_test, y_train, y_test = X[:1600], X[1600:], y[:1600], y[1600:]
    cv1 = TfidfVectorizer(ngram_range=(1,2), binary=True, stop_words='english')
    X_train_cv1 = cv1.fit_transform(X_train)
    X_test_cv1  = cv1.transform(X_test)

    lr = LogisticRegression()
    lr.fit(X_train_cv1, y_train)
    y_pred_cv1 = lr.predict(X_test_cv1)
    st.text("")
    st.text("")
    st.text("")
    st.text('Model Report:\n ' + classification_report(y_test, y_pred_cv1))
    cm = confusion_matrix(y_test, y_pred_cv1)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'],
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':10}, cmap="YlGnBu")

    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_cv1)
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


def NB1_tfidf(data) :

    data = data.tail(2000)
    data["sentiment"] = data["sentiment"].replace({"positive":1,"negative":0})
    X = data.review_text
    y = data.sentiment

    X_train, X_test, y_train, y_test = X[:1600], X[1600:], y[:1600], y[1600:]
    cv1 = TfidfVectorizer(stop_words='english')
    X_train_cv1 = cv1.fit_transform(X_train)
    X_test_cv1  = cv1.transform(X_test)

    mnb = MultinomialNB()
    mnb.fit(X_train_cv1, y_train)
    y_pred_cv1 = mnb.predict(X_test_cv1)
    st.text("")
    st.text("")
    st.text("")
    st.text('Model Report:\n ' + classification_report(y_test, y_pred_cv1))
    cm = confusion_matrix(y_test, y_pred_cv1)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'],
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':10}, cmap="YlGnBu")

    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_cv1)
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


def NB2_tfidf(data) :

    data = data.tail(2000)
    data["sentiment"] = data["sentiment"].replace({"positive":1,"negative":0})
    X = data.review_text
    y = data.sentiment

    X_train, X_test, y_train, y_test = X[:1600], X[1600:], y[:1600], y[1600:]
    cv1 = TfidfVectorizer(ngram_range=(1,2), binary=True, stop_words='english')
    X_train_cv1 = cv1.fit_transform(X_train)
    X_test_cv1  = cv1.transform(X_test)

    bnb = BernoulliNB()
    bnb.fit(X_train_cv1, y_train)
    y_pred_cv1 = bnb.predict(X_test_cv1)
    st.text("")
    st.text("")
    st.text("")
    st.text('Model Report:\n ' + classification_report(y_test, y_pred_cv1))
    cm = confusion_matrix(y_test, y_pred_cv1)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'],
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':10}, cmap="YlGnBu")

    st.pyplot()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_cv1)
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


if (model_results == False) & (review == False) :

    st.subheader("About Data")
    st.text("")

    st.write("""
    ##### The IMDB Spoiler Dataset

    This dataset is collected from IMDB. It contains meta-data about items as well as user reviews with information
    regarding whether a review contains a spoiler or not. For more details on the attributes, please check file descriptions.
    Following stats provide a good sense of the scale of the dataset:

    • records = 573913

    • users = 263407

    • movies = 1572

    • spoiler reviews = 150924

    • users with at least one spoiler review = 79039

    • items with at least one spoiler review = 1570


    """)

    st.text("")

    if st.button('👉🏻 Data Source 👈🏻'):
        js = "window.open('https://www.kaggle.com/rmisra/imdb-spoiler-dataset')"  # New tab or window
        js = "window.location.href = 'https://www.kaggle.com/rmisra/imdb-spoiler-dataset'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)


    st.text("")
    st.text("")
    st.text("")
    st.subheader("Reviews Rating Distribution")
    fig1
    st.subheader("Reviews Spoiler Distribution")
    fig2
    fig3
    st.subheader("Reviews Sentiment Distribution")
    fig4
    st.pyplot()



elif (model_results == True) & (review == False) :

    if (nlp_technique == "Count Vectorizer") & (ml_algorithm == "LogisticRegression-Unigram") :
        LogReg1(df)

    elif (nlp_technique == "Count Vectorizer") & (ml_algorithm == "LogisticRegression-Bigram") :
        LogReg2(df)

    elif (nlp_technique == "Count Vectorizer") & (ml_algorithm == "MultinomialNB") :
        NB1(df)

    elif (nlp_technique == "Count Vectorizer") & (ml_algorithm == "BernoulliNB") :
        NB2(df)

    elif (nlp_technique == "TF-IDF Vectorizer") & (ml_algorithm == "LogisticRegression-Unigram") :
        LogReg1_tfidf(df)

    elif (nlp_technique == "TF-IDF Vectorizer") & (ml_algorithm == "LogisticRegression-Bigram") :
        LogReg2_tfidf(df)

    elif (nlp_technique == "TF-IDF Vectorizer") & (ml_algorithm == "MultinomialNB") :
        NB1_tfidf(df)

    elif (nlp_technique == "TF-IDF Vectorizer") & (ml_algorithm == "BernoulliNB") :
        NB2_tfidf(df)



elif (model_results == False) & (review == True) :
    comment = st.text_input('Add your comment', '')
    X = df.review_text
    y = df.sentiment
    # The second document-term matrix has both unigrams and bigrams, and indicators instead of counts
    cv2 = CountVectorizer(ngram_range=(1,2), binary=True, stop_words='english')

    X_train_cv2 = cv2.fit_transform(X)
    X_test_cv2  = cv2.transform([comment])

    lr = LogisticRegression()

    # Train the  model
    lr.fit(X_train_cv2, y)
    y_pred_cv2 = lr.predict(X_test_cv2)

    st.write('Your comment is : ', y_pred_cv2[0].upper())
