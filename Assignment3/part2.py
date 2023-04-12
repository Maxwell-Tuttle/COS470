import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import FastText
import random
from sklearn.model_selection import train_test_split
from post_parser_record import PostParserRecord



def read_tsv_test_data(file_path):
    # Takes in the file path for test file and generate a dictionary
    # of question id as the key and the list of question ids similar to it
    # as value. It also returns the list of all question ids that have
    # at least one similar question
    dic_similar_questions = {}
    lst_all_test = []
    with open(file_path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            question_id = int(row[0])
            lst_similar = row[1] #list(map(int, row[1:]))
            dic_similar_questions[question_id] = int(lst_similar)
            lst_all_test.append(question_id)
            lst_all_test.extend(lst_similar)
    return dic_similar_questions, lst_all_test


def get_sentence_embedding(model, sentence):
    # This method takes in the trained model and the input sentence
    # and returns the embedding of the sentence as the average embedding
    # of its words
    words = sentence.split(" ")
    vector = model.wv[words[0]]
    for i in range(1, len(words)):
        vector = vector + model.wv[words[i]]
    return vector / len(words)


dic_similar_questions, lst_all_test = read_tsv_test_data("duplicate_questions.tsv")


#model = FastText.load("newFastText.model")

post_file = "Posts_law (3).xml"

post_reader = PostParserRecord(post_file)

all_questions = []
for question_id in post_reader.map_questions:
    all_questions.append(question_id)

positive_pairs = [list(ele) for ele in list(dic_similar_questions.items())]


print(positive_pairs)

negative_pairs = []
for i in range(len(positive_pairs)):
    # Randomly select a question that is not in the positive sample
    negative_question = random.choice(all_questions)
    # print(negative_question)

    # Build the negative pair
    print(positive_pairs[i][0])
    negative_pair = [positive_pairs[i][0], negative_question]

    negative_pairs.append(negative_pair)

print(negative_pairs)

all_pairs = positive_pairs + negative_pairs

print(all_pairs)

train_df, test_df = train_test_split(all_pairs, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

print(train_df)
print(test_df)
print(val_df)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(500, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


net = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(50):
    running_loss = 0.0
    for i in range(len(train_df)):
        q1 = train_df[i][0]
        q2 = train_df[i][0]

