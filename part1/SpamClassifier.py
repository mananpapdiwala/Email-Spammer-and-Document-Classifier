import sys
import time
import os
import re


class FilesData:
    # Variables to store number of times a word appeared in all the spam/non spam files
    spam_words_f = {}
    n_spam_words_f = {}

    # Variables to store number probability of a word apperaing in spam/non spam files considering freq. of word
    p_spam_words_f = {}
    p_n_spam_words_f = {}

    # Variable that store no. of files that has the particular word
    no_of_files_with_this_word_in_spam = {}
    no_of_files_with_this_word_in_n_spam = {}

    # Variable that store probability of word apperaing in a document
    p_spam_words_01 = {}
    p_n_spam_words_01 = {}

    # Variables to store file count
    spam_files_count = 0
    n_spam_files_count = 0

    # Variables to store word count
    total_words_spam_files = 0
    total_words_n_spam_files = 0

    # Variables that store file as a word frequency list
    spam_files = []
    n_spam_files = []

    # List of words
    words = []

    # Probability Table  {word: [p being spam considering freq, p being not spam considering freq, p being spam
    # considering how many spam files have this word, p being not spam considering how many spam files have this word]}
    probability_table = {}


def getFileList(x_dataset):
    spam_file_list = os.listdir(x_dataset + "/spam")
    non_spam_file_list = os.listdir(x_dataset + "/notspam")
    return {"spam": spam_file_list, "notspam": non_spam_file_list}


def getFileData(x_all_emails_list, x_dataset):
    spam_files = [{}, 0]
    not_spam_files = [{}, 0]

    fd.spam_files_count = len(x_all_emails_list["spam"])
    fd.n_spam_files_count = len(x_all_emails_list["notspam"])

    for file_name in x_all_emails_list["spam"]:
        spam_files = readFile(x_dataset + "/spam/" + file_name, spam_files, "spam")

    fd.total_words_spam_files = spam_files[1]
    fd.spam_words_f = spam_files[0]

    for file_name in x_all_emails_list["notspam"]:
        not_spam_files = readFile(x_dataset + "/notspam/" + file_name, not_spam_files, "notspam")

    fd.total_words_n_spam_files = not_spam_files[1]
    fd.n_spam_words_f = not_spam_files[0]

    return {"spam": spam_files, "notspam": not_spam_files}


def readFile(file_name, current_data, filetype):
    words_in_this_file = {}
    words = current_data[0]
    total_no_of_words = 0
    file_data = open(file_name, 'r')

    for line in file_data:
        for word in line.split():
            if str(re.search('[a-zA-Z]', word)) == "None":
                continue
            word = re.sub('[^a-zA-Z0-9@]', '', word)
            word = word.lower()
            total_no_of_words += 1

            if not (word in fd.words):
                fd.words.append(word)

            if word in words:
                words[word] += 1
            else:
                words[word] = 1

            if word in words_in_this_file:
                words_in_this_file[word] += 1
            else:
                if filetype == "spam":
                    if word in fd.no_of_files_with_this_word_in_spam:
                        fd.no_of_files_with_this_word_in_spam[word] += 1
                    else:
                        fd.no_of_files_with_this_word_in_spam[word] = 1
                else:
                    if word in fd.no_of_files_with_this_word_in_n_spam:
                        fd.no_of_files_with_this_word_in_n_spam[word] += 1
                    else:
                        fd.no_of_files_with_this_word_in_n_spam[word] = 1
                words_in_this_file[word] = 1
    if filetype == "spam":
        fd.spam_files.append(words_in_this_file)
    else:
        fd.n_spam_files.append(words_in_this_file)
    return [words, current_data[1] + total_no_of_words]


def getProbabilityDistribution(words):
    for word in words["spam"][0]:
        fd.p_spam_words_f[word] = (1.0 * fd.spam_words_f[word]) / fd.total_words_spam_files

    for word in words["notspam"][0]:
        fd.p_n_spam_words_f[word] = (1.0 * fd.n_spam_words_f[word]) / fd.total_words_n_spam_files

    return [fd.p_spam_words_f, fd.p_n_spam_words_f]


def getProbabilityDistribution2(p_if_not_present_f, p_if_not_present_01):

    for word in fd.words:
        fd.probability_table[word] = [p_if_not_present_f, p_if_not_present_f, p_if_not_present_01, p_if_not_present_01]
        if word in fd.no_of_files_with_this_word_in_spam:
            fd.probability_table[word][0] = (1.0 * fd.spam_words_f[word]) / fd.total_words_spam_files
            fd.probability_table[word][2] = (1.0 * fd.no_of_files_with_this_word_in_spam[word]) / fd.spam_files_count

        if word in fd.no_of_files_with_this_word_in_n_spam:
            fd.probability_table[word][1] = (1.0 * fd.n_spam_words_f[word]) / fd.total_words_n_spam_files
            fd.probability_table[word][3] = (1.0 * fd.no_of_files_with_this_word_in_n_spam[word]) / fd.n_spam_files_count

    for word in fd.probability_table:
        print str(word) + " " + str(fd.probability_table[word])

    return 0


def writeProbabilityInFile(filePath, probabilityIfNotPresent):
    file = open(filePath, 'w')
    for word in fd.p_spam_words_f:
        file.write('%s,%s' % (word, fd.p_spam_words_f[word]))
        file.write(' ')
    file.write('\n')
    for word in fd.p_n_spam_words_f:
        file.write('%s,%s' % (word, fd.p_n_spam_words_f[word]))
        file.write(' ')
    file.close()


def writeProbabilityInFile2(filePath, probabilityIfNotPresent):
    file = open(filePath, 'w')
    for word in fd.p_spam_words_f:
        file.write('%s,%s' % (word, fd.p_spam_words_f[word]))
        file.write(' ')
    file.write('\n')
    for word in fd.p_n_spam_words_f:
        file.write('%s,%s' % (word, fd.p_n_spam_words_f[word]))
        file.write(' ')
    file.close()


start_time = time.time()
print time.asctime(time.localtime(time.time()))

(mode, technique, dataset, modelfile) = sys.argv[1:]

fd = FilesData()

if mode == "train":
    email_directories = getFileList(dataset)
    word_set = getFileData(email_directories, dataset)
    getProbabilityDistribution(word_set)
    getProbabilityDistribution2(0.0001, 0.0002)
    writeProbabilityInFile(modelfile, 0.00001)

end_time = time.time()
print time.asctime(time.localtime(time.time()))
print end_time - start_time
