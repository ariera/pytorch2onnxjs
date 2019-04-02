#%%

#%%
pale_blue_dot="""
We succeeded in taking that picture [from deep space], and, if you look at it, you see a dot. That's here. That's home. That's us. On it, everyone you ever heard of, every human being who ever lived, lived out their lives. The aggregate of all our joys and sufferings, thousands of confident religions, ideologies and economic doctrines, every hunter and forager, every hero and coward, every creator and destroyer of civilizations, every king and peasant, every young couple in love, every hopeful child, every mother and father, every inventor and explorer, every teacher of morals, every corrupt politician, every superstar, every supreme leader, every saint and sinner in the history of our species, lived there on a mote of dust, suspended in a sunbeam.

The Earth is a very small stage in a vast cosmic arena. Think of the rivers of blood spilled by all those generals and emperors so that in glory and in triumph they could become the momentary masters of a fraction of a dot. Think of the endless cruelties visited by the inhabitants of one corner of the dot on scarcely distinguishable inhabitants of some other corner of the dot. How frequent their misunderstandings, how eager they are to kill one another, how fervent their hatreds. Our posturings, our imagined self-importance, the delusion that we have some privileged position in the universe, are challenged by this point of pale light.

Our planet is a lonely speck in the great enveloping cosmic dark. In our obscurity – in all this vastness – there is no hint that help will come from elsewhere to save us from ourselves. It is up to us. It's been said that astronomy is a humbling, and I might add, a character-building experience. To my mind, there is perhaps no better demonstration of the folly of human conceits than this distant image of our tiny world. To me, it underscores our responsibility to deal more kindly and compassionately with one another and to preserve and cherish that pale blue dot, the only home we've ever known
."""

# %%
import numpy as np
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
class TextPreprocesor:
    def __init__(self, text):
        self.tokens = nltk.word_tokenize(text)
        self.vocabulary_size = len(set(self.tokens))
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        integer_encoded = self.label_encoder.fit_transform(np.array(self.tokens))
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        self.onehot_encoder.fit(integer_encoded)
    def encode(self, token):
        integer_code = self.label_encoder.transform([token])
        integer_code = integer_code.reshape(len(integer_code), 1)
        onehot_code = self.onehot_encoder.transform(integer_code)
        return onehot_code
    def decode(self, onehot_code):
        integer_code = self.onehot_encoder.inverse_transform(onehot_code)
        interger_code = int(integer_code[0][0])
        # interger_code = np.argmax(integer_code)
        return self.label_encoder.inverse_transform([interger_code])[0]
    def get_bigrams(self):
        tokens = self.tokens
        bigrams_left2right = [(tokens[index], tokens[index+1]) for index in range(len(tokens) - 1)]
        bigrams_right2left = [(tokens[index], tokens[index-1]) for index in range(len(tokens) - 1, 0, -1)]
        return bigrams_left2right + bigrams_right2left
    def get_encoded_bigrams(self):
        bigrams = self.get_bigrams()
        encode_bigram = lambda x: (self.encode(x[0]), self.encode(x[1]))
        return list(map(encode_bigram, bigrams))

#%%
t=TextPreprocesor(pale_blue_dot)
word = "planet"
code = t.encode(word)
assert t.decode(code) == word
# t.get_bigrams()
t.get_encoded_bigrams()

#%%
