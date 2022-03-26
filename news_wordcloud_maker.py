import spacy
from newsapi import NewsApiClient
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import date, timedelta
from collections import Counter
import pickle





# Load spacy and prepare newsapi
nlp_eng = spacy.load('en_core_web_lg')
api_key = ''
with open('API_KEY','r') as file:
    api_key = file.readlines()[0]
newsapi = NewsApiClient (api_key=api_key)

# Retrieve articles in the past x days
today = date.today()
delta = timedelta(5)
articles = []
for i in range(1, 6):
    articles.extend(newsapi.get_everything(q='coronavirus', language='en', from_param=str(today - delta), to=str(today), sort_by='relevancy', page=i)['articles'])


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
# print(cleaned_articles)

df = pd.DataFrame(cleaned_articles)
df = df.dropna()
df.head()

def get_keywords_eng(text):
    doc = nlp_eng(text)
    result = []
    pos_tag = ['VERB', 'NOUN', 'PROPN']
    for token in doc:
        if (token.pos_ in pos_tag):
            result.append(token.text)
    return result

results = []
for content in df.content.values:
    results.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])

df['keywords'] = results

df.head()



# Get most common 100 words across all articles, not 5 per article
results = []
for content in df.content.values:
    results.extend([word.lower() for word in get_keywords_eng(content)])
temps = str([word for word, count in Counter(results).most_common(100)])


# Create word cloud
text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()