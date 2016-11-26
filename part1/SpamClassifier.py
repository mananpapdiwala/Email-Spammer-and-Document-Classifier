import sys
import time
import os
import re


class FilesData:
    # Variables to store number of times a word appeared in all the spam/non spam files
    spam_words_f = {}
    n_spam_words_f = {}

    # Variables to store number probability of a word appearing in spam/non spam files considering freq. of word
    p_spam_words_f = {}
    p_n_spam_words_f = {}

    # Variable that store no. of files that has the particular word
    no_of_files_with_this_word_in_spam = {}
    no_of_files_with_this_word_in_n_spam = {}

    # Variable that store probability of word appearing in a document
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
    words = {}

    # Actual word for the word
    word_list_with_actual_word = {}

class ProbailityTable:
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
    i = 0
    for file_name in x_all_emails_list["spam"]:
        spam_files = readFile(x_dataset + "/spam/" + file_name, spam_files, "spam")

    fd.total_words_spam_files = spam_files[1]
    fd.spam_words_f = spam_files[0]

    for file_name in x_all_emails_list["notspam"]:
        not_spam_files = readFile(x_dataset + "/notspam/" + file_name, not_spam_files, "notspam")

    fd.total_words_n_spam_files = not_spam_files[1]
    fd.n_spam_words_f = not_spam_files[0]

    return {"spam": spam_files, "notspam": not_spam_files}

# Regex for amount Reference: http://stackoverflow.com/questions/2150205/can-somebody-explain-a-money-regex-that-just-checks-if-the-value-matches-some-pa
# Regex for email http://stackoverflow.com/questions/8022530/python-check-for-valid-email-address
def readFile(file_name, current_data, filetype):
    words_in_this_file = {}
    words = current_data[0]
    total_no_of_words = 0
    file_data = open(file_name, 'r')

    for line in file_data:
        for word in line.split():
            actual_word = word

            if word.find("$") <> -1:
                word = "dollar"

            if not word.isalpha() and word.find("@") == -1 and word.find(".com") == -1 and word.find(".net") == -1:
                continue

            word = re.sub('[^a-zA-Z0-9]', '', word)

            if word == '':
                continue

            word = word.lower()

            total_no_of_words += 1

            if not (word in fd.words):
                fd.words[word] = 1
                fd.word_list_with_actual_word[word] = actual_word
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


def getProbabilityDistribution_old(words):
    for word in words["spam"][0]:
        fd.p_spam_words_f[word] = (1.0 * fd.spam_words_f[word]) / fd.total_words_spam_files

    for word in words["notspam"][0]:
        fd.p_n_spam_words_f[word] = (1.0 * fd.n_spam_words_f[word]) / fd.total_words_n_spam_files

    return [fd.p_spam_words_f, fd.p_n_spam_words_f]


def getProbabilityDistribution():
    p_if_not_present_f = 1.0/fd.total_words_spam_files
    p_if_not_present_01 = 1.0/fd.spam_files_count
    for word in fd.words:
        pt.probability_table[word] = [p_if_not_present_f, p_if_not_present_f, p_if_not_present_01, p_if_not_present_01]
        if word in fd.no_of_files_with_this_word_in_spam:
            pt.probability_table[word][0] = (1.0 * fd.spam_words_f[word]) / fd.total_words_spam_files
            pt.probability_table[word][2] = (1.0 * fd.no_of_files_with_this_word_in_spam[word]) / fd.spam_files_count

        if word in fd.no_of_files_with_this_word_in_n_spam:
            pt.probability_table[word][1] = (1.0 * fd.n_spam_words_f[word]) / fd.total_words_n_spam_files
            pt.probability_table[word][3] = (1.0 * fd.no_of_files_with_this_word_in_n_spam[word]) / fd.n_spam_files_count
    return 0


def writeProbabilityInFile_old(filePath):
    file = open(filePath, 'w')
    for word in fd.p_spam_words_f:
        file.write('%s,%s' % (word, fd.p_spam_words_f[word]))
        file.write(' ')
    file.write('\n')
    for word in fd.p_n_spam_words_f:
        file.write('%s,%s' % (word, fd.p_n_spam_words_f[word]))
        file.write(' ')
    file.close()


def writeModel(filePath):
    output_file = open(filePath, 'w')
    for word in pt.probability_table:
        output_file.write('%s %s ' % (word, pt.probability_table[word][0]))
        output_file.write('%s %s ' % (pt.probability_table[word][1], pt.probability_table[word][2]))
        output_file.write('%s\n' % (pt.probability_table[word][3]))
    output_file.close()


def readModel(filePath):
    input_file = open(filePath, 'r')
    for line in input_file:
        words = line.split()
        pt.probability_table[words[0]] = [float(words[1]), float(words[2]), float(words[3]), float(words[4])]
    input_file.close()


def findPGivenWords(x_files):
    result = [0, 0, 0, 0]
    for mail in x_files:
        probability_01 = 1.0
        probability_f = 1.0
        for word in mail:
            if word in pt.probability_table:
                probability_f *= pt.probability_table[word][0]/pt.probability_table[word][1]
                probability_01 *= pt.probability_table[word][2] / pt.probability_table[word][3]

        if probability_f > 1.0:
            result[0] += 1
        else:
            result[1] += 1

        if probability_01 > 1.0:
            result[2] += 1
        else:
            result[3] += 1

    return result


def findMostProbableWords(x, y):
    words = [""]*20
    probability = [0.0]*20
    for word in pt.probability_table:
        p_of_word = pt.probability_table[word][x]/pt.probability_table[word][y]
        min_index = 0
        min_probability = 10 ** 100
        for i in range(len(probability)):
            if probability[i] < min_probability:
                min_probability = probability[i]
                min_index = i
        if p_of_word > min_probability:
            words[min_index] = word
            probability[min_index] = p_of_word
    return words


def printfMostProbableWords(words):
    result = []
    for word in words:
        result.append(fd.word_list_with_actual_word[word])
    return result

###################################
start_time = time.time()
print time.asctime(time.localtime(time.time()))
print ""
###################################


(mode, technique, dataset, modelfile) = sys.argv[1:]

fd = FilesData()
pt = ProbailityTable()

if mode == "train":
    email_directories = getFileList(dataset)
    getFileData(email_directories, dataset)
    getProbabilityDistribution()
    writeModel(modelfile)

    print ""
    words = findMostProbableWords(0, 1)
    print "P(S=1|w) considering f:" + str(printfMostProbableWords(words))
    words = findMostProbableWords(2, 3)
    print "P(S=1|w) considering presense of word in spam:" + str(printfMostProbableWords(words))
    words = findMostProbableWords(1, 0)
    print "P(S=0|w) considering f:" + str(printfMostProbableWords(words))
    words = findMostProbableWords(3, 2)
    print "P(S=0|w) considering presense of word in spam:" + str(printfMostProbableWords(words))


if mode == "test":
    readModel(modelfile)
    email_directories = getFileList(dataset)
    getFileData(email_directories, dataset)
    result1 = findPGivenWords(fd.spam_files)
    result2 = findPGivenWords(fd.n_spam_files)
    print "ConfusionMatrix where index 0 = Spam and index 1 = Not Spam, row i column j show the number "
    print "of test exemplars whose correct label is i, but that were classified as j"

    confusionMatrix_f = [[result1[0], result1[1]],[result2[0], result2[1]]]
    confusionMatrix_01 = [[result1[2], result1[3]], [result2[2], result2[3]]]

    print "Considering f: " + str(confusionMatrix_f)
    print "Considering presence of word: " + str(confusionMatrix_01)


#######################
end_time = time.time()
print ""
print time.asctime(time.localtime(time.time()))
print end_time - start_time