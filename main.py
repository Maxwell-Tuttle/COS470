from gensim.models import FastText
from scipy import spatial
import csv
from post_parser_record import PostParserRecord


def get_sentence_embedding(model, sentence):
    # This method takes in the trained model and the input sentence
    # and returns the embedding of the sentence as the average embedding
    # of its words
    words = sentence.split(" ")
    vector = model.wv[words[0]]
    for i in range(1, len(words)):
        vector = vector + model.wv[words[i]]
    return vector / len(words)


sampleTexts = ["This is example1", "This is example two", "This is example three"]
# There are parameters here that you should define
#model = FastText(vector_size=3, window=2, min_n=1)
#model.build_vocab(sampleTexts)

# training the model
#model.train(sampleTexts, total_examples=len(sampleTexts), epochs=10)

# saving the model in-case you need to reuse it
#model.save("fastText.model")

#vec1 = get_sentence_embedding(model, sampleTexts[0])
#vec2 = get_sentence_embedding(model, sampleTexts[1])
#vec3 = get_sentence_embedding(model, sampleTexts[2])

# calculating cosine similarity
#result = 1 - spatial.distance.cosine(vec1, vec2)
#print(result)

#result = 1 - spatial.distance.cosine(vec1, vec3)
#print(result)


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
            lst_similar = list(map(int, row[1:]))
            dic_similar_questions[question_id] = lst_similar
            lst_all_test.append(question_id)
            lst_all_test.extend(lst_similar)
    return dic_similar_questions, lst_all_test


def train_model(lst_sentences):
    # training
    # Your code
    new_model = FastText(vector_size=500, window=5, min_n=1, sg=1)
    new_model.build_vocab(lst_sentences)

    new_model.train(lst_sentences, total_examples=len(lst_sentences), epochs=10)

    return new_model


def main():
    duplicate_file = "duplicate_questions.tsv"
    post_file = "Posts_law (3).xml"
    dic_similar_questions, lst_all_test = read_tsv_test_data(duplicate_file)

    post_reader = PostParserRecord(post_file)

    lst_training_sentences = []
    lst_training_answers = []
    for question_id in post_reader.map_questions:
        if question_id in lst_all_test:
            continue
        question = post_reader.map_questions[question_id]
        title = question.title
        body = question.body
        # Collect sentences here
        lst_training_sentences.append(title + body)

        lst_answers = question.answers
        if lst_answers is not None:
            for answer in lst_answers:
                answer_body = answer.body
                # Collection sentences here
                # Your code
                lst_training_answers.append(answer_body)

    # train your model
    new_model = train_model(lst_training_sentences)
    new_model = train_model(lst_training_answers)

    new_model.save("newFastText.model")

    new_model = FastText.load("newFastText.model")

    # This dictionary will have the test question id as the key
    # and the most similar question id as the value
    dictionary_result_title = {}
    dictionary_result_body = {}
    ques_title_dic = {}
    ques_body_dic = {}

    for question_id in post_reader.map_questions:
        question = post_reader.map_questions[question_id]
        ques_title = question.title
        ques_body = question.body

        # Your code
        ques_title_dic[question_id] = get_sentence_embedding(new_model, ques_title)
        ques_body_dic[question_id] = get_sentence_embedding(new_model, ques_body)




    # finding Similar questions using fastText model
    for test_question_id in dic_similar_questions:
        # This gets the title and body from the test question

        # for this question you have to find the similar questions
        test_question = post_reader.map_questions[test_question_id]
        test_title = test_question.title
        test_body = test_question.body

        test_title = get_sentence_embedding(new_model, test_title)
        test_body = get_sentence_embedding(new_model, test_body)

        most_similar_body = 0
        temp_id_body = 0
        most_similar_title = 0
        temp_id_title = 0
        for question_id in post_reader.map_questions:
            # we are not comparing a question with itself
            if question_id == test_question_id:
                continue

            # This gets the question that we are comparing to
            ques_title = ques_title_dic[question_id]
            ques_body = ques_body_dic[question_id]

            # use your model and calculate the cosine similarity between the questions
            # save the question id with the highest cosine similarity

            # Your code
            # calculating cosine similarity
            result_title = 1 - spatial.distance.cosine(test_title, ques_title)
            if most_similar_title < result_title:
                most_similar_title = result_title
                temp_id_title = question_id

            result_body = 1 - spatial.distance.cosine(test_body, ques_body)
            if most_similar_body < result_body:
                most_similar_body = result_body
                temp_id_body = question_id

            dictionary_result_body[test_question_id] = temp_id_body
            dictionary_result_title[test_question_id] = temp_id_title

    print(dictionary_result_body)
    print(dictionary_result_title)



    # Calculate average P@1 and print it.
    # Your code
    p_body = 0
    p_title = 0
    for test_question_id in dic_similar_questions:
        if dictionary_result_body[test_question_id] == dic_similar_questions[test_question_id][0]:
            p_body = p_body + 1

        if dictionary_result_title[test_question_id] == dic_similar_questions[test_question_id][0]:
            p_title = p_title + 1

    p_body_total = p_body / len(dic_similar_questions)
    p_title_total = p_title / len(dic_similar_questions)

    print(p_body_total)
    print(p_title_total)

main()

