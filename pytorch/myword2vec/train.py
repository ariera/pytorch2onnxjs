#%%
import torch
from torch import nn, optim

from pytorch.myword2vec.data import TextPreprocesor, pale_blue_dot
#%%
class MyWord2VecNet(nn.Module):
    def __init__(self, vocabulary_size=5, embedding_size=3):
        super(MyWord2VecNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(vocabulary_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, vocabulary_size)
        )
    def forward(self, x):
        return self.network(x)

#%%
def train(epochs=1000):
    model = MyWord2VecNet(vocabulary_size=5, embedding_size=3)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    t = TextPreprocesor(pale_blue_dot)

    for epoch in range(epochs):
        total_loss = 0
        for (x, y) in t.get_encoded_bigrams():
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            model.zero_grad()
            y_hat = model(x)
            loss = loss_function(y_hat, y)
            # loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch: {epoch}\tloss: {total_loss}")
if __name__ == "__main__":
    train()

# %%
t = TextPreprocesor(pale_blue_dot)
print(len(set(t.tokens)))
print(t.vocabulary_size)
model = MyWord2VecNet(vocabulary_size=t.vocabulary_size, embedding_size=3)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
encoded_bigrams = t.get_encoded_bigrams()
x, y = encoded_bigrams[0]

x = torch.from_numpy(x[0]).float()
y = torch.from_numpy(y[0]).float()
x
model.zero_grad()
y_hat = model(x)
y_hat
loss = loss_function(y_hat, y.reshape((1,t.vocabulary_size)))
