import re
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

patriots_df = pd.read_csv("PatriotsTweets.csv", sep=",", header=0)
cowboys_df = pd.read_csv("CowboysTweets.csv", sep=",", header=0)
vikings_df = pd.read_csv("VikingsTweets.csv", sep=",", header=0)


# Clean the data
def cleanTxt(text):
    text = re.sub('https?:\/\/t.co/[\w]+', "", text)
    text = re.sub('@\w+', "", text)

    return text.lower()


def getScores(df):
    pos_list = []
    neg_list = []
    neu_list = []

    for i in range(len(df)):
        text = cleanTxt(df.iloc[i]["Tweet"])
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        # Print labels and scores
        for j in range(scores.shape[0]):
            label = config.id2label[ranking[j]]
            score = scores[ranking[j]]
            if label == "negative":
                neg_list.append(score)
            elif label == "positive":
                pos_list.append(score)
            else:
                neu_list.append(score)

    df["Positive"] = pos_list
    df["Negative"] = neg_list
    df["Neutral"] = neu_list

    sentiment_list = []

    for k in range(len(pos_list)):
        pos = pos_list[k]
        neg = neg_list[k]
        neu = neu_list[k]

        if (pos > neg) & (pos > neu):
            sentiment_list.append("positive")
        elif (neg > pos) & (neg > neu):
            sentiment_list.append("negative")
        else:
            sentiment_list.append("neutral")

    df["Automated Sentiment"] = sentiment_list
    return df


def getAccuracy(df):
    count = 0
    for i in range(len(df)):
        if df.iloc[i]["Manual Sentiment"] == df.iloc[i]["Automated Sentiment"]:
            count += 1
        else:
            continue

    accuracy = count / len(df)

    return accuracy

def getNegativeScore(df):
    count = 0
    for i in range(len(df)):
        count += df.iloc[i]["Negative"]

    total_negativity = count / len(df)
    return total_negativity


print(getScores(patriots_df).to_string())
print(getScores(cowboys_df).to_string())
print(getScores(vikings_df).to_string())

print("New England Patriots accuracy: " + str(getAccuracy(patriots_df)))
print("Dallas Cowboys accuracy: " + str(getAccuracy(cowboys_df)))
print("Minnesota Vikings accuracy: " + str(getAccuracy(vikings_df)))

print("New England Patriots total negativity: " + str(getNegativeScore(patriots_df)))
print("Dallas Cowboys total negativity: " + str(getNegativeScore(cowboys_df)))
print("Minnesota Vikings total negativity: " + str(getNegativeScore(vikings_df)))