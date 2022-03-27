import spacy
from newsapi import NewsApiClient
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import date, timedelta
from collections import Counter
import pickle


# Get keywords of info
def get_keywords_eng(text):
    doc = nlp_eng(text)
    result = []
    pos_tag = ['VERB', 'NOUN', 'PROPN']
    for token in doc:
        if token.pos_ in pos_tag:
            result.append(token.text)
    return result

# Load spacy and prepare newsapi
nlp_eng = spacy.load('en_core_web_lg')
api_key = ''
with open('API_KEY','r') as file:
    api_key = file.readlines()[0]
newsapi = NewsApiClient (api_key=api_key)

# Retrieve articles in the past x days
end_date = date.today()
start_date = end_date - timedelta(25)
articles = newsapi.get_everything(q='coronavirus', language='en', from_param=str(start_date), to=str(end_date), sort_by='relevancy', page_size=100)['articles']

# Store data locally
filename = 'articlesCOVID.pckl'
pickle.dump(articles, open(filename, 'wb'))

filename = 'articlesCOVID.pckl'
loaded_model = pickle.load(open(filename, 'rb'))

filepath = 'articlesCOVID.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))


# Clean data, keep only data we are interested in
cleaned_articles = []
for article in articles:
    cleaned_articles.append({'title': article['title'], 'date': article['publishedAt'], 'desc': article['description'], 'content': article['content']})

# Transform data into data frame
df = pd.DataFrame(cleaned_articles)
df = df.dropna()

# Get 5 most common words per article
top_words_per_article = []
for content in df.content.values:
    top_words_per_article.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])

# Add common words to new column in data frame
df['keywords'] = top_words_per_article
df.to_excel('Data.xlsx')

# Get most common words across all articles, not just 5 per article
all_keywords = []
for content in df.content.values:
    all_keywords.extend([word.lower() for word in get_keywords_eng(content)])
keywords_ranked = str([word for word, count in Counter(all_keywords).most_common()])

# Create word cloud
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(keywords_ranked)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()