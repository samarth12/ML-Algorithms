import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import re
import collections
import nltk
import scipy as scipy
#from statistics import mean
import matplotlib.pyplot as plt

#Clean data, remove stopword and other unwanted characters
#Returns the cleaned data, dictionary of the vocabulary and all the unique words on the vocabulary
def clean_data(train_data):
    with open("stopwords.txt", 'r') as f:
        stopwords = f.read().split("\n")
    w_punc = nltk.WordPunctTokenizer()
    stopwords.append('br')
    stopwords = [re.sub(r'[^\w\s\d]', '', sw.lower()) for sw in stopwords]
    vocab_dict = []
    vocab_set = set()
    for idx, review in enumerate(train_data):
        review = review.decode('utf-8')
        review = review.replace('\n', ' ')
        review = re.sub(" \d+", " ", review)
        pattern = r"[{}]".format("-?!,.;:/<>'\(\)\"\"")
        review = re.sub(pattern, " ", review)
        review = review.lower()
        review = review.strip()
        tokens = w_punc.tokenize(review)
        filtered_tokens = [token for token in tokens if token not in stopwords]
        review = ' '.join(filtered_tokens)
        train_data[idx] = review
        word_list = review.split(" ")
        word_dict = {}
        for w in word_list:
            vocab_set.add(w)
            word_dict[w] = review.count(w)

        vocab_dict.append(word_dict)
    return train_data, vocab_dict, vocab_set


#Calculate the conditional probability of each word given the class
#Returns the probability for a word given a class
def conditional_prob(word, word_dict, label_count):
    if word not in word_dict or word_dict[word] < 1:
        probability = 1 / (len(word_dict.keys()) + label_count)
    else:
        probability = word_dict[word] / label_count
    return probability


#Calculate the final probability for the input for both the classes
#Returns the final probability for the two classes
def calculate_final_probability(review, pos_dict, neg_dict, count_pos, count_neg, prior_pos, prior_neg):
    word_tokens = review.split(" ")
    probability_pos = 1
    probability_neg = 1
    for word in word_tokens:
        probability_pos = probability_pos * conditional_prob(word, pos_dict, count_pos)
        probability_neg = probability_neg * conditional_prob(word, neg_dict, count_neg)

    final_prob_pos = probability_pos * prior_pos
    final_prob_neg = probability_neg * prior_neg

    return final_prob_pos, final_prob_neg


#Test the input, if the right class is predicted or not and calculate the accuracy
#Returns the accuracy of the model
def evaluate_accuracy(x_test, y_test,  pos_dict, neg_dict, count_pos, count_neg,
                      prior_pos, prior_neg):
    true_postives = 0
    false_positives = 0

    for idx, single_test in enumerate(x_test):
        probability_pos, probability_neg = calculate_final_probability(single_test, pos_dict, neg_dict, count_pos,
                                                                       count_neg, prior_pos, prior_neg)
        if y_test[idx] == 0 and probability_pos >= probability_neg:
            true_postives = true_postives + 1
        elif y_test[idx] == 1 and probability_pos < probability_neg:
            true_postives = true_postives + 1
        else:
            false_positives = false_positives + 1

    accuracy = true_postives / (true_postives + false_positives)
    return accuracy


#Returns the triplet, a dictionary for positive and negative counts, positive and negative counts
def generate_triplets_counts(vocabulary_list, X_train_file_word_dict, y_train, pos_dict, neg_dict, bow_matrix,
                             flag_train_data):
    triple_tuple = []
    count_pos = 0
    count_neg = 0
    if flag_train_data:
        for key, value in enumerate(vocabulary_list):
            #print("Working magic on ", key + 1, " record")
            for idx, word_dictionary in enumerate(X_train_file_word_dict):
                if value in word_dictionary.keys():
                    bow_matrix[key][idx] = word_dictionary[value]
                    triple_tuple.append([key, idx, word_dictionary[value]])

                    if y_train[idx] == 1:
                        neg_dict[value] = neg_dict[value] + word_dictionary[value]
                        count_neg = count_neg + 1
                    else:
                        pos_dict[value] = pos_dict[value] + word_dictionary[value]
                        count_pos = count_pos + 1

        return triple_tuple, count_pos, count_neg, pos_dict, neg_dict


#Load and process data
def data_process(test_fraction):
    folders = ['pos', 'neg']
    movie_reviews = sklearn.datasets.load_files('.', 'movie reviews data', folders, decode_error=True)
    x_train, x_test, y_train, y_test = train_test_split(movie_reviews.data, movie_reviews.target,
                                                        test_size=test_fraction)
    x_test = movie_reviews.data
    y_test = movie_reviews.target

    x_train, x_train_vocabulary_dict, x_vocabulary_set = clean_data(x_train)
    x_test, x_test_vocabulary_dict, x_test_vocabulary = clean_data(x_test)
    vocabulary_list = np.array(list(x_vocabulary_set))
    count_train = len(y_train)
    count_prior = collections.Counter(y_train)
    prior_pos = count_prior[0] / count_train
    prior_neg = count_prior[1] / count_train

    vocab_size = len(vocabulary_list)
    bow_matrix = np.zeros((vocab_size, count_train))

    pos_dict = dict()
    neg_dict = dict()
    for key in vocabulary_list:
        pos_dict[key] = 0
        neg_dict[key] = 0


    flag_train = True
    triple_tuple, count_pos, count_neg, pos_dict_up, neg_dict_up = generate_triplets_counts(vocabulary_list,
                                                                                            x_train_vocabulary_dict,
                                                                                            y_train, pos_dict, neg_dict,
                                                                                            bow_matrix,
                                                                                            flag_train)  # flag to train data
    scipy.io.savemat('submit_matrix', {"output": triple_tuple})

    accuracy = evaluate_accuracy(x_test, y_test, pos_dict_up, neg_dict_up,
                                 count_pos, count_neg, prior_pos, prior_neg)
    print("Accuracy = ", accuracy * 100)
    print("RUN COMPLETE!")
    return accuracy * 100

#One iteration for training and testing
def iterations():
    fractions = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    accuracy_list = []

    for fraction in fractions:
        test_fraction = 1 - fraction
        acc = data_process(test_fraction)
        accuracy_list.append(acc)

    return accuracy_list
    # plt.plot(fractions, accuracy_list, 'ro')
    # plt.axis([0, 1, 50, 100])
    # plt.show()


#START POINT
def entry_point():
    fractions = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    final_list = []
    for i in range(5):
        accuracy_list = iterations()
        final_list.append(accuracy_list)

    average_list = np.mean(final_list, axis=0)
    plt.plot(fractions, average_list, 'ro')
    plt.axis([0, 1, 50, 100])
    plt.show()

if __name__ == "__main__":
    entry_point()