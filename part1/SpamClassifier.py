import sys
import time
import os
import re


def getFileList(x_dataset):
    spam_file_list = os.listdir(x_dataset + "/spam")
    non_spam_file_list = os.listdir(x_dataset + "/notspam")
    return {"spam": spam_file_list, "notspam": non_spam_file_list}


def getFileData(x_all_emails_list, x_dataset):
    # i = 0
    spam_files = [{}, 0]
    not_spam_files = [{}, 0]
    for file_name in x_all_emails_list["spam"]:
        spam_files = readFile(x_dataset + "/spam/" + file_name, spam_files)
        # i += 1

    # i = 0
    for file_name in x_all_emails_list["notspam"]:
        not_spam_files = readFile(x_dataset + "/notspam/" + file_name, not_spam_files)
        # i += 1
    return {"spam": spam_files, "notspam": not_spam_files}


def readFile(file_name, current_data):
    words = current_data[0]
    total_no_of_words = 0
    file_data = open(file_name, 'r')

    for line in file_data:
        for word in line.split():
            if str(re.search('[a-zA-Z]', word)) == "None":
                continue
            word = re.sub('[^a-zA-Z0-9@]', '', word)
            total_no_of_words += 1
            if words.has_key(word):
                words[word] += 1
            else:
                words[word] = 1
    return [words, current_data[1] + total_no_of_words]


def getBayseinProbabilityTable(x_word_freq_in_files_and_total_words, x_all_emails_list):
    no_of_words_in_spam_emails = x_word_freq_in_files_and_total_words["spam"][1]
    no_of_words_in_not_spam_emails = x_word_freq_in_files_and_total_words["notspam"][1]
    total_words = no_of_words_in_spam_emails + no_of_words_in_not_spam_emails
    probability_of_word_in_spam_at0_not_spam_at1 = {}               # {word: [probability in spam, probability in not spam]}
    for word in x_word_freq_in_files_and_total_words["spam"][0]:
        probability_of_word_in_spam_at0_not_spam_at1[word] = [(1.0*x_word_freq_in_files_and_total_words["spam"][0][word])/(1.0*no_of_words_in_spam_emails), 0.0000001]

    for word in x_word_freq_in_files_and_total_words["notspam"][0]:
        if probability_of_word_in_spam_at0_not_spam_at1.has_key(word):
            probability_of_word_in_spam_at0_not_spam_at1[word] = [probability_of_word_in_spam_at0_not_spam_at1[word][0],
                                                                  (1.0 * x_word_freq_in_files_and_total_words["notspam"]
                                                                  [0][word]) / (1.0 * no_of_words_in_spam_emails)]

        else:
            probability_of_word_in_spam_at0_not_spam_at1[word] = [0.0000001,
                                                                  (1.0 *
                                                                   x_word_freq_in_files_and_total_words["notspam"][0][
                                                                       word]) / (1.0 * no_of_words_in_spam_emails)]

    return probability_of_word_in_spam_at0_not_spam_at1


start_time = time.time()
print time.asctime(time.localtime(time.time()))

(mode, technique, dataset, modelfile) = sys.argv[1:]

all_emails_list = getFileList(dataset)
word_freq_in_files_and_total_words = getFileData(all_emails_list, dataset)
probability_of_word_in_spam_at0_not_spam_at1 = getBayseinProbabilityTable(word_freq_in_files_and_total_words, all_emails_list)
print probability_of_word_in_spam_at0_not_spam_at1["money"]

end_time = time.time()
print time.asctime(time.localtime(time.time()))
print end_time - start_time