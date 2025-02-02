from textblob import TextBlob
from newspaper import Article
'''
import nltk
print(nltk.data.find('tokenizers/punkt'))
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

text = "This is a test. Let's check if tokenization works."
print(sent_tokenize(text))
'''

def get_text_from_Articles(url):
    article = Article(url)

    article.download()
    article.parse()
    article.nlp()
    print(article.title)
    # text = article.text
    return article.summary

def find_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity #-1 to 1

    print(f'sentiment: {sentiment}')

'''
find_sentiment(url="https://en.wikipedia.org/wiki/Mathematics")
find_sentiment(url="https://www.bbc.com/news/articles/cd9qzg92g72o")
# findsentiment(url="https://www.latimes.com/environment/story/2025-01-31/california-trump-water-pumping")
'''
with open('myText2.txt', 'r') as f:
    text = f.read()

find_sentiment(text)