'''
File: train_model.py
Description: This file contains all code needed to train a character-based LTSM model based on the Syriac corpus.
'''

# Import modules
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# Model class definition
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x


# Function for training the model
def train_model(training_filename, model_filename):
    # load ascii text and covert to lowercase
    raw_text = open(training_filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    
    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    
    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100    # number of characters per sample
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
    X = X / float(n_vocab)
    y = torch.tensor(dataY)
    print(X.shape, y.shape)

    # model creation
    n_epochs = 20   # number of times the model passes through the whole dataset
    batch_size = 50     # number of samples per batch
    model = CharModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)
    best_model = None
    best_loss = np.inf

    # model training
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                y_pred = model(X_batch)
                loss += loss_fn(y_pred, y_batch)
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()
            print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

    # save model into provided model file
    torch.save([best_model, char_to_int], model_filename)


# Train the model
# train_model("training.txt", "single-char.pth")