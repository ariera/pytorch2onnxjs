import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class Word2VecNet(nn.Module):
    def __init__(self, vocab_size, embedding_dims):
        super(Word2VecNet, self).__init__()
        # vocab_size = 10
        # embedding_dims = 3
        self.embedding_layer = nn.Linear(vocab_size, embedding_dims)
        self.output_layer    = nn.Linear(embedding_dims, vocab_size)
    def forward(self, x):
        x = self.embedding_layer(x).clamp(min=0)
        x = self.output_layer(x)
        # return F.log_softmax(x, dim=0)
        return x




def get_input_layer(word_idx, vocabulary_size):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

def train(idx_pairs, vocabulary_size, embedding_dims):
    model = Word2VecNet(vocabulary_size, embedding_dims)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 100
    for epoch in range(num_epochs):
        for data, target in idx_pairs:
            y_true = torch.from_numpy(np.array([target])).long()
            # ===================forward=====================
            output = model(get_input_layer(data, vocabulary_size))
            import pdb; pdb.set_trace()
            loss = criterion(output, y_true)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.data[0]))

def main():
    corpus = [
        'he is a king',
        'she is a queen',
        'he is a man',
        'she is a woman',
        'warsaw is poland capital',
        'berlin is germany capital',
        'paris is france capital',
    ]
    tokenized_corpus = [x.split() for x in corpus]
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    vocabulary_size = len(vocabulary)
    window_size = 2
    idx_pairs = []
    # for each sentence
    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))
    idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array
    embedding_dims = 5
    return train(idx_pairs, vocabulary_size, embedding_dims)

if __name__ == "__main__":
    print("let's go!")
    main()
