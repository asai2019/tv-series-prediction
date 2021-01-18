# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 17:47:15 2021
Helper functions to complete TV series classification tasks. 
@author: The Prince
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense


def plot_outcome_distribution(labels,prevalences,colors = ['#ff9999','#66b3ff','#99ff99']):
    """custom plotting function for outcome distribution"""
    fig, ax = plt.subplots(figsize=(6,6))
    wedges, texts, autotexts = ax.pie(prevalences, labels=labels, colors=colors, 
                                      labeldistance=None,
                                      shadow=True, autopct='%.2f%%', startangle=90)
    ax.axis('equal')
    ax.legend(wedges,labels)
    plt.setp(autotexts, size=12)
    plt.title('TV Series Duration Outcome Distribution')
    plt.show()


def remove_numbers(text):
    """custom NLP function to remove numbers"""
    return "".join([word for word in text if not word.isdigit()])

def remove_punctuation(text):
    """custom NLP function to remove the punctuation"""
    return text.translate(str.maketrans('', '', string.punctuation))

def stem_words(text):
    """custom NLP function to stem words"""
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])

def remove_stopwords(text):
    """custom NLP function to remove the stopwords"""
    STOPWORDS = stopwords.words('english') + stopwords.words('spanish') + stopwords.words('german')
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def preprocess_corpus(corpus):
    """custom function to preprocess corpora"""
    corpus = corpus.str.lower() # Make lowercase
    corpus = corpus.apply(lambda text: remove_numbers(text))
    corpus = corpus.apply(lambda text: remove_punctuation(text))
    corpus = corpus.apply(lambda text: remove_stopwords(text))
    corpus = corpus.apply(lambda text: stem_words(text))
    return corpus

def combine_sparse_data(df,dtm):
    """custom function to append BOW features to original feature set"""
    return hstack([csr_matrix(df.values), dtm])

def report_test_performance_multiclass(classifier,X_test,y_test,name='Logistic Regression'):
    """custom function to report test set metrics and plot confusion matrix"""
    # Report test accuracy for optimal model
    prob = classifier.predict_proba(X_test)
    pred = np.argmax(prob,axis=-1)
    print('{0} Test AUC: {1:.4f}'.format(name,roc_auc_score(y_test, prob, average='weighted', multi_class='ovr')))
    print('{0} Test Balanced Accuracy: {1:.4f}'.format(name,balanced_accuracy_score(y_test, pred)))

    cm = confusion_matrix(y_test, pred)

    fig1, ax1 = plt.subplots()
    sns.set_style('darkgrid')
    ax = sns.heatmap(cm,annot=True, fmt="d")
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    ax.set_title('{} Multiclass Confusion Matrix'.format(name))
    
def report_test_performance_binary(classifier,X_test,y_test,name='Logistic Regression'):
    """custom function to report test set metrics and plot confusion matrix"""
    # Report test accuracy for optimal model
    prob = classifier.predict_proba(X_test)
    if prob.shape[1] == 1:
        pred = prob
    else:
        prob = prob[:,1]
    pred = (prob>0.5).astype(int)
    print('{0} Test AUC: {1:.4f}'.format(name,roc_auc_score(y_test, prob)))
    print('{0} Test Balanced Accuracy: {1:.4f}'.format(name,balanced_accuracy_score(y_test, pred)))

    cm = confusion_matrix(y_test, pred)

    fig1, ax1 = plt.subplots()
    sns.set_style('darkgrid')
    ax = sns.heatmap(cm,annot=True, fmt="d")
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    ax.set_title('{} Multiclass Confusion Matrix'.format(name))    

def plot_coefficients(coefs):
    fig, axes = plt.subplots(figsize=(12,8),nrows=2,ncols=2)
    axes[0,0].set_title('Coeficients for Short Term Prediction')
    axes[0,1].set_title('Coeficients for Medium Term Prediction')
    axes[1,0].set_title('Coeficients for Long Term Prediction')
    short_term_coef = coefs.loc[coefs.coef0.abs().nlargest(20).index,'coef0'].sort_values()
    short_term_coef.plot(kind='barh', color=np.where(short_term_coef > 0, 'g', 'r'), ax=axes[0,0])
    medium_term_coef = coefs.loc[coefs.coef1.abs().nlargest(20).index,'coef1'].sort_values()
    medium_term_coef.plot(kind='barh', color=np.where(medium_term_coef > 0, 'g', 'r'), ax=axes[0,1])
    long_term_coef = coefs.loc[coefs.coef2.abs().nlargest(20).index,'coef2'].sort_values()
    long_term_coef.plot(kind='barh', color=np.where(long_term_coef > 0, 'g', 'r'), ax=axes[1,0])
        
    
def plot_probabilities(prob,labels):
    """custom function to plot predicted probabilities from model"""
    dis = sns.displot(data=prob, kind='kde')
    plt.title('TV Series Predicted Probability Distributions by Category')
    dis._legend.set_title('TV Series Duration Category')
    for t, l in zip(dis._legend.texts, labels): t.set_text(l)
    

def ANNBinaryClassifierModel():
    classifier = Sequential()    
    classifier.add(Dense(input_dim=1024, units=512, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=32, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=16, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=8, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=4, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(activation='sigmoid', units=1))
    return classifier

def ANNMultiClassifierModel():
    classifier = Sequential()    
    classifier.add(Dense(input_dim=1024, units=512, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=32, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=16, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=8, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=4, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dense(activation='softmax', units=3))
    return classifier