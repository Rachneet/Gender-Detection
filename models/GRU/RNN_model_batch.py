# a batch implementation of basic RNN model

from torch.utils import data
from torch.nn.utils import rnn
from sklearn.metrics import accuracy_score

import string
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import preprocess as pp
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dataset(data.Dataset):
    def __init__(self, reviews, labels, lengths):
        self.reviews = reviews
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self,index):
        X = torch.tensor(self.reviews[index],dtype=torch.long)
        y = torch.tensor(self.labels[index],dtype=torch.long)
        s_lengths = self.lengths[index]
        return X,y,s_lengths

class DataInference(data.Dataset):
    def __init__(self, reviews, lengths):
        self.reviews = reviews
        self.lengths = lengths

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self,index):
        X = torch.tensor(self.reviews[index],dtype=torch.long)
        s_lengths = self.lengths[index]
        return X,s_lengths


class Encoder(nn.Module):
    def __init__(self, input_size, encoding_size, hidden_size, output_size, layers, padding_idx):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.layers = layers
        self.embedding = nn.Embedding(input_size, encoding_size, padding_idx=padding_idx)
        self.e2i = nn.Linear(encoding_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=self.layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.batch_first = True

    def forward(self, X, X_lengths, batch_size):

        self.hidden = self.initHidden(batch_size)
        X = self.embedding(X)
        X = self.e2i(X)
        X = rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        X, self.hidden = self.gru(X, self.hidden)

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        idx = (torch.cuda.LongTensor(X_lengths) - 1).view(-1, 1).expand(len(X_lengths), X.size(2))

        time_dimension = 1 if self.batch_first else 0
        idx = idx.unsqueeze(time_dimension)
        X = X.gather(time_dimension, Variable(idx)).squeeze(time_dimension)

        X = self.out(X)
        X = self.sigmoid(X)

        return X

    def initHidden(self,batch_size):
         return torch.zeros(self.layers, batch_size, self.hidden_size).to(device)


def sortbylength(X,y,s_lengths):
    sorted_lengths, indices = torch.sort(s_lengths,descending=True)
    return X[torch.LongTensor(indices),:],y[torch.LongTensor(indices)],sorted_lengths


def train(encoder, dataset_train, dataset_validate, batch_size, epochs=15, learning_rate=0.001):
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss() # cross entropy loss in pytorch combines logsoftmax and negativeloglikelihoodloss...softmax layer not needed
    validation_accuracy = 0
    for i in range(epochs):
        batch_cnt = 0
        loader_train = data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
        for X,y,X_lengths in loader_train:
            batch_cnt+=1
            X,y,X_lengths = sortbylength(X,y,X_lengths)
            X,y,X_lengths = X.to(device),y.to(device),X_lengths.to(device)
            b_size = y.size(0)
            output = encoder(X,X_lengths,b_size)
            loss = criterion(output,y)
            if batch_cnt%100==0:
                print('epochs: ',i)
                print('batch_cnt: ',batch_cnt)
                print('Loss: ',loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print("training complete for epoch: ",i)
        print("Loss after epoch: ",loss)
        print("------------------------")


        loader_validate = data.DataLoader(dataset_validate,batch_size=batch_size)
        accuracy = []
        for X,y,X_lengths in loader_validate:
            X,y,X_lengths = sortbylength(X,y,X_lengths)
            X,y,X_lengths = X.to(device),y.to(device),X_lengths.to(device)
            b_size = y.size(0)
            output = encoder(X,X_lengths,b_size)
            output = F.softmax(output,dim=1)
            value,labels = torch.max(output,1)

            accuracy.append(accuracy_score(y.cpu().numpy(),labels.cpu().numpy()))

        mean_accuracy = np.mean(accuracy)
        print('accuracy: {} at epoch: {}'.format(mean_accuracy,i))
        if validation_accuracy < mean_accuracy:
            validation_accuracy = mean_accuracy
            print('accuracy: ',mean_accuracy)
            torch.save(encoder.state_dict(), 'encoder_model_2.pt')
    

def inference(encoder,dataset_test,batch_size):
    loader = data.DataLoader(dataset_test,batch_size=batch_size)
    for X,y,X_lengths in loader_validate:
        X,y,X_lengths = sortbylength(X,y,X_lengths)
        X,y,X_lengths = X.to(device),y.to(device),X_lengths.to(device)
        b_size = y.size(0)
        output = encoder(X,X_lengths,b_size)
        output = F.softmax(output,dim=1)
        value,labels = torch.max(output,1)

        return accuracy_score(y.cpu().numpy(),labels.cpu().numpy()),y.cpu().numpy(),labels.cpu().numpy()




def sentence2tensor(sentence,w2i,pad,sent_length):
    S = [w2i[w] for w in sentence]
    if len(S)>sent_length:
        return S[:sent_length]
    else:
        x = len(S)
        S = S + [pad for i in range(sent_length - x)]
        return S


def encodeDataset(fname,w2i,padding_idx,sent_length):
    reviews = []
    labels = []
    lengths = []
    count = 0

    with open(fname+'_filtered') as fs:    
         for line in fs:
             count+=1
             label = line[0]
             review = line[2:]
             words = word_tokenize(review.strip())
             if len(words)>0:
                reviews.append(sentence2tensor(words,w2i,padding_idx,sent_length))
                if len(words)>sent_length:
                    lengths.append(sent_length)
                else:
                    lengths.append(len(words))
                labels.append(int(label))
                if count%100000==0:
                    print('Encoded reviews: ',count)


    return reviews,labels,lengths


if __name__=='__main__':
    sent_length = 100
    train_file,validation_file = '../Data/train_s.csv','../Data/test_s.csv'
    #w2i = pp.obtainW2i(sent_length,train = train_file,validate = validation_file,test=test_file)
    #w2i = pp.word2index(train = '../Data/train.csv_filtered',validation = '../Data/validation.csv_filtered',test='../Data/test.csv_filtered')
    #print('Loaded vocabulary')
    #w2i['<PAD>'] = 0

    #vocab_size = len(w2i)

    
    #with open('word2index','wb') as ft:
    #    pickle.dump(w2i,ft)

    with open('word2index','rb') as fs:
        w2i = pickle.load(fs)

    print('loaded vocabulary')
    print('size of vocabulary: ',len(w2i))

    vocab_size = len(w2i)
    padding_idx = 0 

    reviews_train,labels_train,lengths_train = encodeDataset(train_file,w2i,padding_idx,sent_length)
    reviews_validate, labels_validate, lengths_validate = encodeDataset(validation_file,w2i,padding_idx,sent_length)
    #reviews_test, labels_test, lengths_test = encodeDataset(test_file,w2i,padding_idx,sent_length)


    print('created batches from data loader')


    #print(reviews[2])
    w2i = {}
    dataset_train = Dataset(reviews_train,labels_train,lengths_train)
    dataset_validate = Dataset(reviews_validate,labels_validate,lengths_validate)
    encoding_size = 50

    hidden_size = 250
    input_size = vocab_size
    output_size = 2
    layers = 2
    batch_size = 256
    encoder = Encoder(input_size, encoding_size, hidden_size, output_size,layers, padding_idx)
    encoder = encoder.to(device)
    train(encoder,dataset_train, dataset_validate, batch_size)
    

