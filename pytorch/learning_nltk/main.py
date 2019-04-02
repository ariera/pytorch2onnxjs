#%%
import nltk

#%%
# nltk.download()

# %%
from nltk.book import text6

dist = nltk.FreqDist(text6)
len(dist)
vocab1 = dist.keys()
vocab1
#%%
nltk.word_tokenize(" ".join(text6))
#%%
WNLemma = nltk.WordNetLemmatizer()
[WNLemma.lemmatize(t) for t in vocab1]

