from builtins import range
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
stop = set(stopwords.words('english'))
stop.update(('#$%','$$$'))
def my_tokenizer(token):
    #s = s.lower() # downcase
    #tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    token = [t for t in token if len(t) > 2] # remove short words, they're probably not useful
    token = [wordnet_lemmatizer.lemmatize(t) for t in token] # put words into base form
    token = [t for t in token if t not in stop] # remove stopwords
    return token
pos_reviews = []
neg_reviews = []
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    #print (words)
    tokens = my_tokenizer(words)
    positive_tokenized.append(tokens)
    for t in tokens:
        if t not in word_index_map:
            word_index_map[t] = current_index
            current_index += 1
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    tokens = my_tokenizer(words)
    negative_tokenized.append(tokens)
    for t in tokens:
        if t not in word_index_map:
            word_index_map[t] = current_index
            current_index += 1
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1

# shuffle the data and create train/test splits
# try it multiple times!
np.random.shuffle(data)

X = data[:,:-1]
Y = data[:,-1]

# last 100 rows will be test
Xtrain = X[:1700,]
Ytrain = Y[:1700,]
Xtest = X[1700:,]
Ytest = Y[1700:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
#print("Classification rate:", model.score(Xtest, Ytest))
from sklearn.metrics import accuracy_score
pred = model.predict(Xtest)
print(accuracy_score(pred,Ytest))            
threshold = 0.3
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)            
