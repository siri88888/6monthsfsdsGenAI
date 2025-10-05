# XML scrapping for xml sheet 

import os
os.chdir(r"C:\Users\ttwrd\Downloads\14. NLP WEB SCRAPING\xml_single articles")

import xml.etree.ElementTree as ET

tree = ET.parse("769952.xml") 
root = tree.getroot()

root=ET.tostring(root, encoding='utf8').decode('utf8')

root

import re, string, unicodedata
import nltk

from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text=re.sub('  ','',text)
    return text

sample = denoise_text(root)
print(sample)
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(sample)
for token in doc:
    print(token.text,":",token.pos_)

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
stopwords = list(STOP_WORDS) 
stopwords
# lets get the tokens from text
tokens = [token.text for token in doc]
print(tokens) 
punctuation # also called as noisy characters
word_frequencies = {}

for word in doc:
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
word_frequencies
#print(word_frequencies)
max_frequency = max(word_frequencies.values())
max_frequency 

for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency
    #print(word_frequencies)
word_frequencies
#this is the normalized frequencies of each wordsentence_tokens = [sent for sent in doc.sents]
sentence_tokens = [sent for sent in doc.sents]
sentence_tokens
len(sentence_tokens)
sentence_scores = {}

for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]
sentence_scores 
from heapq import nlargest       
select_length = int(len(sentence_tokens)*0.4)
select_length
summary = nlargest(select_length,sentence_scores, key = sentence_scores.get)


final_summary = [word.text for word in summary]
final_summary
import nltk
import nltk.corpus
text=" ".join(final_summary)
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
#wordcloud = WordCloud(width=420, height=200, margin=2,background_color='cyan', colormap="Dark2").generate(text)
#wordcloud = WordCloud(width=420, height=200, margin=2,background_color='white',colormap='Accent',mode='RGBA').generate(text)
import numpy as np
from PIL import Image
mask = np.array(Image.open(r'C:\Users\ttwrd\Downloads\geeksimage.png'))
wc = WordCloud(stopwords = STOPWORDS,
               mask = mask, background_color = "white",
               max_words = 2000, max_font_size = 500,
               random_state = 42, width = mask.shape[1],
               height = mask.shape[0])
wc.generate(text)
plt.imshow(wc, interpolation="None")
plt.axis('off')
plt.show()



#plt.imshow(wordcloud, interpolation='quadric',)
"""plt.imshow(wordcloud,  interpolation='bilinear')

plt.axis("off")
plt.margins(x=0, y=0)
plt.show()"""