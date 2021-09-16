## End-to-End Content-Based Message Spam Detection

  In this project, we built, trained, evaluated, and deployed a model that classifies text messages as spam or ham based on the message's content. The term *ham* designates a message that is generally legitimate and is not considered spam.

  The model is built using the Multinomial Naive Bayes (MNB) classifier. It is a fast algorithm that is suited for text classification.
  
  We deployed the model in a web app for online spam detection using Streamlit. It can be accessed at:
  
  https://share.streamlit.io/c-youssef/spam-detection/main/spam_detection_app.py
  
#

![App screenshot](https://github.com/C-Youssef/spam-detection/blob/d3c7d51912348de1c4c6fd381b4c9305a377307d/media/app_screenshot.png)

#

  The Jupyter notebook spam_detection.ipynb is divided into six sections:
  
1.   Loading the labeled SMS dataset. 
2.   Visualizing the data.
3.   Applying natural language processing (NLP) to the textual data.
4.   Extracting numerical features from the textual data using Term Frequency - Inverse Document Frequency (TF-IDF).
5.   Building, training, and evaluating the model.
6.   Conclusions.


### Prerequisites

We used [Python 3](https://www.python.org/) and the following libraries: 

* [Pandas](https://pandas.pydata.org/docs/index.html#): data structures and analysis for python.
* [Matplotlib](https://matplotlib.org/): python plotting library.
* [Wordcloud](https://pypi.org/project/wordcloud/): word cloud generator in Python.
* [NLTK](https://www.nltk.org/): natural language processing.
* [Scikit-learn](scikit-learn.org): machine learning.
* [Streamlit](https://streamlit.io/): Python-based web deployment of data apps.

### 
### Data source

  The data set used in this project is the SMS Spam Collection Data Set. It is a public set of SMS labeled messages that have been collected for mobile phone spam research. 
  
  It is hosted in the machine learning repository of the UCL Center of Machine Learning and Intelligent systems https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection and stored there in a single text file, but also available in a CSV file at https://www.kaggle.com/uciml/sms-spam-collection-dataset.
  