
print("------------------TENSORFLOW-------------------")

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

vocab_size_tf=10000
max_len_tf = 200

(X_train_tf,y_train_tf),(X_test_tf,y_test_tf) = imdb.load_data(num_words=vocab_size_tf)

X_train_tf = pad_sequences(X_train_tf,maxlen=max_len_tf,padding='post')
X_test_tf = pad_sequences(X_test_tf,maxlen=max_len_tf,padding='post')

print(f"Training Data Shape: {X_train_tf.shape}")
print(f"Test Data Shape: {X_test_tf.shape}")

model_tf = Sequential([
    Embedding(input_dim=vocab_size_tf, output_dim=128),
    SimpleRNN(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

model_tf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model_tf.summary()

history_tf = model_tf.fit(X_train_tf,y_train_tf, epochs=5, batch_size=32, validation_split=0.2)

loss_tf, accuracy_tf = model_tf.evaluate(X_test_tf,y_test_tf)

print(f"Test Loss: {loss_tf:.4f}, Test Accuracy: {accuracy_tf:.4f}")






print("------------------PYTORCH-------------------")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size_py = 10000
max_len_py = 200

(X_train_py,y_train_py), (X_test_py,y_test_py) = imdb.load_data(num_words=vocab_size_py)

X_train_py = pad_sequences(X_train_py,maxlen=max_len_py,padding='post')
X_test_py = pad_sequences(X_test_py,maxlen=max_len_py,padding='post')

# Converts numpy arrays into pytorch tensors and creates dataset of paired inputs and labels
train_dataset = TensorDataset(torch.tensor(X_train_py),torch.tensor(y_train_py))
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

class RNNModel(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.rnn = nn.RNN(embedding_dim,hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)
        
        
    def forward(self,x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return torch.sigmoid(self.fc(hidden.squeeze(0)))
    
    
model_py = RNNModel(vocab_size_py, embedding_dim=128, hidden_dim=128, output_dim=1)

criterion = nn.BCELoss()
optimizer = optim.Adam(model_py.parameters(), lr=0.001)

def train_rnn(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze(1)
            loss = criterion(predictions, y_batch.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

train_rnn(model_py, train_loader, criterion, optimizer)

def evaluate_rnn(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test)).squeeze(1)
        loss = criterion(predictions, torch.tensor(y_test).float())
        accuracy = ((predictions>0.5)==torch.tensor(y_test).float()).float().mean()
    print(f"Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")
    
    
evaluate_rnn(model_py, X_test_py, y_test_py)