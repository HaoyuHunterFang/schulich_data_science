# %% [markdown]
# ## BMO

# %%
import pandas as pd

tweets = pd.read_excel('/Users/houhiroshisakai/Desktop/Schulich/Term 2/MBAN 6090/Twitter-BMO.xlsx')

texts = tweets['Content1'].tolist()

# %%
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')
nltk.download('stopwords')

#Define cleaning
def clean_text(text):
    text = text.lower()  # convert to lower case
    text = re.sub(r'\d+', '', text)  # remove number
    text = re.sub(r'http\S+', '', text)  # remove url
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation mark
    tokens = word_tokenize(text)  
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # remove stop words
    return ' '.join(tokens)

clean_texts = [clean_text(text) for text in texts]

# %%
from textblob import TextBlob

# Analyze emotion for each tweet
sentiments = [TextBlob(text).sentiment.polarity for text in clean_texts]

# add sentiment to df
tweets['Sentiment'] = sentiments

# %%
tweets[['Content1', 'Sentiment']].head()

# %%
tweets['Emotion'] = tweets['Sentiment'].apply(lambda x: 'Positive' if x>0 else ('Negative' if x < 0 else 'Neutral'))

# %%
# Latent Dirichlet Allocation
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ContextVectorize
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(clean_texts)

# Apply LDA
lda = LatentDirichletAllocation(n_components=5)
topics = lda.fit_transform(X)

print(lda.components_)

# %%
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

print_top_words(lda, vectorizer.get_feature_names_out(), n_top_words=10)

# %%
# To assign topics to documents, you will look at the highest topic weight in topics
doc_topic = lda.transform(X)
# Then for each document, find the topic with the highest weight
most_dominant_topic = doc_topic.argmax(axis=1)
# Add this to your DataFrame
tweets['Dominant_Topic'] = most_dominant_topic

# %%
from collections import Counter
# Function to identify top n keywords in texts
def get_top_n_words(corpus, n=None, stop_words=None):
    vec = CountVectorizer(stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Define exclude words
exclude_words = ['wealth', 'bmo', 'management', 'financial', 'us', 'business', 'group','report','bank','according','chief','head','market','markets','investment']

top_words = get_top_n_words(clean_texts, n=10, stop_words=exclude_words)

# %%
top_words

# %%
# Group by Emotion and Date and get the average sentiment
grouped_tweets = tweets.groupby(['Emotion', 'Published_Date']).agg({'Sentiment': 'mean'})

# %%
for word, freq in top_words:
    tweets[word] = tweets['Content1'].str.contains(word, case=False)

# %%
tweets['Published_Date'] = pd.date_range(start='2016-08-16', periods=len(tweets), freq='D')

import matplotlib.pyplot as plt
for emotion in ['Positive', 'Negative', 'Neutral']:
    # Filter the tweets for the current emotion
    emotion_tweets = tweets[tweets['Emotion'] == emotion]
    
    # Group by date and calculate the mean sentiment
    date_sentiment = emotion_tweets.groupby('Published_Date').agg({'Sentiment': 'mean'})
    
    # Plot the trend
    plt.plot(date_sentiment.index, date_sentiment['Sentiment'], label=emotion)

# Add some plot details
plt.title('Sentiment Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.legend()
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()

# %%
tweets

# %%
tweets_sorted = tweets.sort_values(by=['Dominant_Topic', 'Published_Date'], ascending=[False, False])

# %%
# Save the current df to new excel file

new_file_path = '/Users/houhiroshisakai/Desktop/Schulich/Term 2/MBAN 6090/BMO_updated.xlsx'

tweets_sorted.to_excel(new_file_path, index=False)

# %% [markdown]
# ## CIBC

# %%
tweets_cibc = pd.read_excel('/Users/houhiroshisakai/Desktop/Schulich/Term 2/MBAN 6090/Twitter-CIBC.xlsx')

texts_cibc = tweets_cibc['Content'].tolist()

# %%
#Define cleaning
def clean_text(texts_cibc):
    texts_cibc = texts_cibc.lower()  # convert to lower case
    texts_cibc = re.sub(r'\d+', '', texts_cibc)  # remove number
    texts_cibc = re.sub(r'http\S+', '', texts_cibc)  # remove url
    texts_cibc = re.sub(r'[^\w\s]', '', texts_cibc)  # remove punctuation mark
    tokens = word_tokenize(texts_cibc)  
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # remove stop words
    return ' '.join(tokens)

clean_texts = [clean_text(texts_cibc) for texts_cibc in texts_cibc]

# %%
# Analyze emotion for each tweet
sentiments_cibc = [TextBlob(texts_cibc).sentiment.polarity for texts_cibc in clean_texts]

# add sentiment to df
tweets_cibc['Sentiment'] = sentiments_cibc

# %%
tweets_cibc[['Content', 'Sentiment']].head()

# %%
tweets_cibc['Emotion'] = tweets_cibc['Sentiment'].apply(lambda x: 'Positive' if x>0 else ('Negative' if x < 0 else 'Neutral'))

# %%
# ContextVectorize
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(clean_texts)

# Apply LDA
lda = LatentDirichletAllocation(n_components=5)
topics = lda.fit_transform(X)

print(lda.components_)

# %%
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

print_top_words(lda, vectorizer.get_feature_names_out(), n_top_words=10)

# %%
# To assign topics to documents, you will look at the highest topic weight in topics
doc_topic = lda.transform(X)
# Then for each document, find the topic with the highest weight
most_dominant_topic_cibc = doc_topic.argmax(axis=1)
# Add this to your DataFrame
tweets_cibc['Dominant_Topic'] = most_dominant_topic_cibc

# %%
tweets_cibc['Published_Time'] = pd.to_datetime(tweets_cibc['Published_Time'], format='%b %d, %Y', errors='coerce')

# %%
tweets_cibc['Published_Time'] = tweets_cibc['Published_Time'].fillna(pd.Timestamp('2023-12-15'))

# %%
from collections import Counter
# Function to identify top n keywords in texts
def get_top_n_words(corpus, n=None, stop_words=None):
    vec = CountVectorizer(stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Define exclude words
exclude_words = ['wealth', 'cibc', 'management', 'financial', 'private', 'business', 'group','us','bank','cibcprivatewealth','chief','director','market','jamie','investment','news','new']

top_words = get_top_n_words(clean_texts, n=10, stop_words=exclude_words)

# %%
top_words

# %%
# Group by Emotion and Date and get the average sentiment
grouped_tweets_cibc = tweets_cibc.groupby(['Emotion', 'Published_Date']).agg({'Sentiment': 'mean'})

# %%
tweets_cibc['Published_Time'] = pd.date_range(start='2016-01-01', periods=len(tweets_cibc), freq='D')

import matplotlib.pyplot as plt
for emotion in ['Positive', 'Negative', 'Neutral']:
    # Filter the tweets for the current emotion
    emotion_tweets = tweets_cibc[tweets_cibc['Emotion'] == emotion]
    
    # Group by date and calculate the mean sentiment
    date_sentiment = emotion_tweets.groupby('Published_Time').agg({'Sentiment': 'mean'})
    
    # Plot the trend
    plt.plot(date_sentiment.index, date_sentiment['Sentiment'], label=emotion)

# Add some plot details
plt.title('Sentiment Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.legend()
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()

# %%
tweets_sorted_cibc = tweets_cibc.sort_values(by=['Dominant_Topic', 'Published_Time'], ascending=[False, False])

# %%
# Save the current df to new excel file

new_file_path_cibc = '/Users/houhiroshisakai/Desktop/Schulich/Term 2/MBAN 6090/CIBC_updated.xlsx'

tweets_sorted_cibc.to_excel(new_file_path_cibc, index=False)

# %% [markdown]
# ## Morgan Stanley

# %%
tweets_ms = pd.read_excel('/Users/houhiroshisakai/Desktop/Schulich/Term 2/MBAN 6090/Twitter-Morgan Stanley Wealth Management-2.xlsx')

texts_ms = tweets_ms['Content1'].tolist()

# %%
#Define cleaning
def clean_text(texts_ms):
    texts_ms = texts_ms.lower()  # convert to lower case
    texts_ms = re.sub(r'\d+', '', texts_ms)  # remove number
    texts_ms = re.sub(r'http\S+', '', texts_ms)  # remove url
    texts_ms = re.sub(r'[^\w\s]', '', texts_ms)  # remove punctuation mark
    tokens = word_tokenize(texts_ms)  
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # remove stop words
    return ' '.join(tokens)

clean_texts = [clean_text(texts_ms) for texts_ms in texts_ms]

# %%
# Analyze emotion for each tweet
sentiments_ms = [TextBlob(texts_ms).sentiment.polarity for texts_ms in clean_texts]

# add sentiment to df
tweets_ms['Sentiment'] = sentiments_ms

# %%
tweets_ms[['Content1', 'Sentiment']].head()

# %%
tweets_ms['Emotion'] = tweets_ms['Sentiment'].apply(lambda x: 'Positive' if x>0 else ('Negative' if x < 0 else 'Neutral'))

# %%
# ContextVectorize
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(clean_texts)

# Apply LDA
lda = LatentDirichletAllocation(n_components=5)
topics = lda.fit_transform(X)

print(lda.components_)

# %%
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

print_top_words(lda, vectorizer.get_feature_names_out(), n_top_words=10)

# %%
# To assign topics to documents, you will look at the highest topic weight in topics
doc_topic = lda.transform(X)
# Then for each document, find the topic with the highest weight
most_dominant_topic_ms = doc_topic.argmax(axis=1)
# Add this to your DataFrame
tweets_ms['Dominant_Topic'] = most_dominant_topic_ms

# %%
tweets_sorted_ms = tweets_ms.sort_values(by=['Dominant_Topic', 'Published_Date'], ascending=[False, False])

# %%
# Save the current df to new excel file

new_file_path_ms = '/Users/houhiroshisakai/Desktop/Schulich/Term 2/MBAN 6090/MorganStanley_updated.xlsx'

tweets_sorted_ms.to_excel(new_file_path_ms, index=False)

# %%



