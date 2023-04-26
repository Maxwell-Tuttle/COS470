import csv
import random
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, util, models, evaluation
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
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


dic_similar_questions, lst_all_test = read_tsv_test_data("duplicate_questions.tsv")

post_file = "Posts_law.xml"

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
    negative_pair = [positive_pairs[i][0], negative_question]

    negative_pairs.append(negative_pair)

print(negative_pairs)

all_pairs = positive_pairs + negative_pairs

print(all_pairs)

train_df, test_df = train_test_split(all_pairs, test_size=0.1, random_state=42)

print(len(train_df))
print(len(test_df))

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

with torch.no_grad():
    pos_inputs = tokenizer(positive_pairs, padding=True, truncation=True, return_tensors="pt")
    pos_embeddings = model(**pos_inputs).pooler_output
    pos_sim_score = np.inner(pos_embeddings, pos_embeddings)

    neg_inputs = tokenizer(negative_pairs, padding=True, truncation=True, return_tensors="pt")
    neg_embeddings = model(**neg_inputs).pooler_output
    neg_sim_score = np.inner(neg_embeddings, neg_embeddings)

    pos_mean_score = np.mean(pos_sim_score)
    neg_mean_score = np.mean(neg_sim_score)

print(pos_sim_score)
print(neg_sim_score)
