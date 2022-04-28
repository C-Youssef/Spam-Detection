#!/usr/bin/env python
# coding: utf-8
 
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Main function start ****************************************************************************************************************

def main():

    global vectorizer, binomial_NB_model

    _max_width_()
    local_css("icon.css")

# App Title

    st.markdown("<h1> <font color=indigo> Online SPAM Detector </font></h1>", unsafe_allow_html=True)

# App headers

    st.header('Using natural language processing and machine learning for content-based detection of spam messages.')
    
    st.write("")
    
# Show help if needed

    with st.expander("Need help using the demo?"):
        '''
        1. Select a text message on the left sidebar. You may also edit the message or write your own below.
        2. Click on *scan*.
        * From the left sidebar,  you can also display the SMS dataset word clouds. Do you recognize some of the words that we often get in spam?!
        '''
    st.write("")
# Test text messages

    text_list=[
    """ It's good, we'll find a way.""",
    """ Congrats! 1 year special cinema pass for 2 is yours. call 09061209465 now! C Suprman V, Matrix3, StarWars3, etc all 4 FREE! bx420-ip4-5we. 150pm. Dont miss out!""",
    """ Awesome, lemme know whenever you're around""",
    """ You might want to pull out more just in case and just plan on not spending it if you can, I don't have much confidence in derek and taylor's money management""",
    """ Claim a 200 shopping spree, just call 08717895698 now! Have you won! MobStoreQuiz10ppm""",
    """ U still going to the mall?""",
    """ YOUR CHANCE TO BE ON A REALITY FANTASY SHOW call now = 08707509020 Just 20p per min NTT Ltd, PO Box 1327 Croydon CR9 5WB 0870 is a national = rate call""",
    """ Finished class. Where are you?""",
    """important information 4 orange user . today is your lucky day!2find out why log onto http://www.urawinner.com THERE'S A FANTASTIC SURPRISE AWAITING YOU!"""]

# Names of the used classification models

    model_name = ['Multinomial Naive Bayes - scikit']


# Sidebar content start **********************************************************************************************************

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")

    st.sidebar.subheader(' Machine Learning Model')
    selected_model = st.sidebar.selectbox("", model_name)

    st.sidebar.write("")

# Model details

    with st.sidebar.expander("Model details"):
            st.write('[Multinomial Naive Bayes] (https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes) (MNB) classification using [Scikit learn](https://scikit-learn.org). Text preprocessing was done using [NLTK](https://www.nltk.org/). F1-score = 0.92')

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")

# Text message selection

    st.sidebar.subheader(' Message Selection')
    selected_message = st.sidebar.selectbox('', text_list)

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")


# Displaying word clouds of the spam and ham (non-spam) messages

    st.sidebar.subheader(' SMS Dataset Word Clouds')
    wc_spam_display = st.sidebar.checkbox("Spam messages", False)
    wc_ham_display = st.sidebar.checkbox("Non-Spam messages", False)

    st.sidebar.write("")
    st.sidebar.write('*Scroll down if media is not visible*')


    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")

    with st.sidebar.expander("SMS dataset source"):
        st.write('The SMS Spam Collection Dataset was used to train and evaluate the model deployed in this web app. It is a public set of SMS labeled messages that have been collected for mobile phone spam research and hosted in the [machine learning repository of the UCL Center of Machine Learning and Intelligent systems] (https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)')

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.subheader("Contact")
    st.sidebar.markdown('''
        #### **Youssef Charfi**
        [Email me] (mailto:"youssef.charfi@gmail.com")''', unsafe_allow_html=True)

# Sidebar content end **********************************************************************************************************


# We use Pickle to load our saved TFIDF victorizer and Binomial NB SPAM classifier (Trained using scikit learn library)

    with st.spinner("Please wait while the classification model is being loaded"):
        vectorizer = pickle.load(
            open("models/tfidf_victorizer_spam_dataset1.pkl","rb"))
        binomial_NB_model = pickle.load(
            open("models/MultinomialNB_spam_classifier1.pkl","rb"))

    st.write("")
    st.write("")

# Displaying (and editing) the selected text message

    st.subheader("Write a message or select one on the left sidebar:")
    if selected_message:
        selected_message = st.text_area("", selected_message)

    button_clicked = st.button("Scan")

# text message classification as spam or ham

    if button_clicked:
        text_class = Is_it_spam(selected_message)
        if text_class == "spam":
            st.write("This message is spam.")
            icon("delete")
        elif text_class == "ham":
            st.write("This message is not spam.")
            icon("mail")
        else:
            st.write("oops, something went wrong. Reload the spam detection model.")

    st.write("")
    st.write("")

    if wc_spam_display:
        st.subheader('Word cloud of spam messages:')
        st.write("")
        st.image("media/wordcloud_spam.png", width = 750)
    if  wc_ham_display:
        st.subheader('Word cloud of non-spam messages:')
        st.write("")
        st.image("media/wordcloud_ham.png", width = 750)



# Main function end ****************************************************************************************************************


def  preprocessing(message):
    
    # Preprocesses a text message before classification. The Natural Language Toolkit (NLTK) is used for stop-word removal and stemming.
    
    nltk.download('stopwords', quiet=True)
    nltk.download('snowball_data', quiet=True)

    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer

    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    split_regex = '[^a-zA-Z0-9,]+'
    message_out = ''
    for word in re.split(split_regex, message):
        word = word.lower()
        word = re.sub(',', '', word)
        if (word not in stop_words)  and (len(word) != 1):
            message_out = ' '.join([message_out, stemmer.stem(word)])
    return message_out


def Is_it_spam(message):
    
    # Classifies a text message into 'spam' or 'ham' using a TFIDF victorizer and Binomial NB SPAM classifier.
    
    processed_message = [preprocessing(message),]
    x_vec =  vectorizer.transform(processed_message)
    pred = binomial_NB_model.predict(x_vec)
    if pred.tolist()[0] == 1:
        return 'spam'
    else:
        return 'ham'


def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )
    

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)
    
if __name__ == '__main__':    
    main()
