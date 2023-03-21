# IMDB Movie Reviews: Sentiment Analysis
of the challenges for Cohort 25 and 26, choosing NLP problem

## Kaggle Project Link

https://www.kaggle.com/code/ashleshkhajbage/imdb-movie-reviews-sentiment-analysis


I think this a very nice project on NLP Sentiment analysis.

The main Challenge here is that I have to predict the number of positive and negative reviews based on sentiments by using different classification models

We have to determine the sentiment of 50000 movie reviews into different classes.


## Procedure 
So we start with doing following –

1.	Data Preparation: The first step is to download the dataset and prepare it for analysis. 
    - I will need to clean the data by removing punctuation, stop words, and HTML tags. (i.e. removing html strips and noise text)
    - I also had should also convert all text to lowercase and tokenize the text into individual words.

2.	Data Exploration: Once the data is prepared, I then explored it to gain insights into the dataset. 
    - Here I created frequency distributions of words and plot them to see which words appear most frequently in positive and negative reviews.

3.	Feature Extraction: The next step is to extract features from the text that can be used to train a machine-learning model. 
    - I can use techniques such as Bag of Words, TF-IDF, or Word Embeddings to represent the text as numerical features.    
    - TF-IDF : - term frequency-inverse document frequency. 
        - Which is a numerical statistic that measures the importance of string representations such as words, phrases and more in a corpus (document).

4.	Model Selection and Training: Later then I select a machine learning model and train it using the extracted features. 
    - Common models for sentiment analysis include Logistic Regression, Naive Bayes, and Support Vector Machines. 
    - I could also use deep learning models such as Recurrent Neural Networks or Convolutional Neural Networks.
    - So I implemented Linear regression Model on two different feature extraction technique (1st one is bag of words and 2nd is TF-IDF) 

5.	Model Evaluation: Once the model was trained, I evaluated its performance on a held-out test set. I used metrics such as accuracy, precision, recall, and F1-score to evaluate the model's performance.

## Results and Classification Reports 
1. **accuracy scores**
    - **Logistic Regression :**  
        - lr_bow_score : 0.7513, 
        - lr_tfidf_score : 0.7506
	- **Linear support vector machines :** 
		- svm_bow_score : 0.5823
		- svm_tfidf_score : 0.5112
	- **Multinomial Naive Bayes :**
		- mnb_bow_score : 0.7514
		- mnb_tfidf_score : 0.7511


##  Classification Reports 
1. **Logistic Regression**
![Logistic Regression](https://github.com/Ashleshk/IMDB-Movie-Reviews---Sentiment-Analysis/blob/main/images/Screenshot%202023-03-20%20at%2010.09.42%20PM.png)

2. **Linear support vector machines**
![SVM](https://github.com/Ashleshk/IMDB-Movie-Reviews---Sentiment-Analysis/blob/main/images/Screenshot%202023-03-20%20at%2010.10.07%20PM.png)

3. **Multinomial Naive Bayes**
![Naive bayes](https://github.com/Ashleshk/IMDB-Movie-Reviews---Sentiment-Analysis/blob/main/images/Screenshot%202023-03-20%20at%2010.10.26%20PM.png)


## Word cloud for positive review words

![Postive word](https://github.com/Ashleshk/IMDB-Movie-Reviews---Sentiment-Analysis/blob/main/images/Screenshot%202023-03-20%20at%2010.10.45%20PM.png)

## Word cloud for negative review words

![negative word](https://github.com/Ashleshk/IMDB-Movie-Reviews---Sentiment-Analysis/blob/main/images/Screenshot%202023-03-20%20at%2010.10.54%20PM.png)


That’s pretty much it.
