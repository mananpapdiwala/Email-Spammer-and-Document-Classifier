# Considered abc@xyz as a single word rather that 2 words, as the accuracy is better this way
# Also, considered http:abc as a word as the accuracy is better
# Assumes pickle can be used. Used to save decision tree as an object
#
#
# Model 1: Word as a binary feature
# Probability for a word appearing in spam mail is calculated by counting number of files in which the word appears in
# spam mails / no of spam mails. Same goes gor non spam mails.
# P(S=1|w) is calculated by dividing P(w|S=1)/P(w|S=0).
#
# Model 2: Word freq. taken into account
# Probability for a word appearing in spam mail is calculated by counting number of times the word appeared in all
# spam mails / no of spam mails. Same goes gor non spam mails.
# P(S=1|w) is calculated by dividing P(w|S=1)/P(w|S=0)
#
# For decision tree we have selected the word which splits the data with least avg disorder and built the tree further
# Model for dt net is saved in a .pkl file. Used pickle to save model(word is counted same as for bayes)
#
# Considered words containing '$' and 'USD' amount as word 'dollardollar'.
# Considered numbers as word 'numbernumber'.

# Model 1 where word is taken as a binary feature works marginally better in bayes net model.
# Model 1 where word is taken as a binary feature works marginally better in decision tree model.
#
# bayes net words best as a spam classifier
# Results attached below

# For printing the 10 words most associated with spam
# We have calculated P(S=1|w)/P(S=0|w) and printed the top 10 words with highest values. As the probability of word
# being in Spam is much more than in not spam
#
# For printing the 10 words least associated with spam
# We have calculated P(S=0|w)/P(S=1|w) and printed the top 10 words with highest values. As the probability of word
# being in Spam is much less than in not spam


# Also, please see def replace_line(line): to understand what we have considered as a word. We have replaced punchuation
# marks and few words then considered words. So hyperlinks will look different in top 10 words.

# Results: DT
""" """
"""
DECISION TREE: word as binary feature

TRAINING:
Node:hits
	Left Node:zzzzlocalhostnetnoteinccom
	Right Node(Leaf): Not Spam
Node:zzzzlocalhostnetnoteinccom
	Left Node:spambayes
	Right Node(Leaf): Not Spam
Node:spambayes
	Left Node:subscription
	Right Node(Leaf): Not Spam
Node:subscription
	Left Node(Leaf): Spam
	Right Node:tm
Node:tm
	Left Node:zzzzilugexamplecom
	Right Node(Leaf): Not Spam

DECISION TREE: word as frequency
Node:tests
	Left Node:httpclickthruonlinecomclickq
	Right Node(Leaf): Not Spam
Node:httpclickthruonlinecomclickq
	Left Node:font
	Right Node(Leaf): Not Spam
Node:font
	Left Node:mv
	Right Node(Leaf): Spam
Node:mv
	Left Node:jul
	Right Node(Leaf): Spam
Node:jul
	Left Node:oct
	Right Node(Leaf): Spam

TEST:
Confusion Matrix
(Model 1: Words as binary features):
            SPAM        NOT SPAM    ACCURACY
SPAM         1180           5       99.58%
NOT SPAM       83        1286       93.94%

Average Accuracy: 96.76%

Confusion Matrix
(Model 2: Words as frequency):
            SPAM        NOT SPAM    ACCURACY
SPAM         1161          24       97.97%
NOT SPAM       73        1296       94.67%

Average Accuracy: 96.32%

"""

# Results: Bayes
"""
TRAINING:
Top 10 words::
    taking words as binary:
		for P(S=1|w):['jm@netnoteinc', 'cpunks@localhost', 'spamdeath', 'mailings', 'jun', 'cypherpunks-forward@ds', 'kr', 'cypherpunks@ds', 'cpunks@hq', 'mortgage', 'hq', 'webmaster@efi', 'frontpage', 'zzzzason', 'cdo', 'yyyy@netnoteinc', 'zzzz@jmason', 'zzzz@spamassassin', 'spamassassin-sightings@lists', 'cn']

		for P(S=0|w):['rpm', 'newsisfree', 'rpm-zzzlist@freshrpms', 'freshrpms', 'rpm-list-admin@freshrpms', ':rpm-list-request@freshrpms', 'hits', 'pine', 'redhat', 'lnx', 'yyyy@example', 'fork@example', 'jm-rpm@jmason', 'encoding', 'rssfeeds@jmason', 'rpm-zzzlist-admin@freshrpms', 'rpm-list@freshrpms', 'rssfeeds@example', 'egwn', ':rpm-zzzlist-request@freshrpms']


	taking frequency in account:

		for P(S=1|w) :['mortgage', 'cn', 'ba', 'cpunks@localhost', 'ssz', 'cypherpunks@ds', 'txt@dogma', 'mailings', 'locust', 'minder', 'insuranceiq', 'jm@netnoteinc', 'zzzz@jmason', 'msonormal', 'yyyy@netnoteinc', 'hq', 'mv', 'bd', 'ptsize', 'zzzzason']

		for P(S=0|w):['yyyy@example', 'zdnet', 'encoding', 'fork@example', 'freshrpms', 'rpm-list@freshrpms', 'hits', 'cnet', 'lockergnome', 'rpm-zzzlist-admin@freshrpms', 'exmh', 'rssfeeds@example', 'rpm-zzzlist@freshrpms', 'iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii', 'rssfeeds@jmason', 'egwn', 'rpm', 'weblogs', 'redhat', 'listman']



TEST
Confusion Matrix
(Model 1: Words as binary features):
            SPAM        NOT SPAM    ACCURACY
SPAM         1175          10       99.16%
NOT SPAM       22        1347       98.39%

Average Accuracy: 98.78%


Confusion Matrix
(Model 2: Words frequency taken into account):
            SPAM        NOT SPAM    ACCURACY
SPAM         1151          34       97.13%
NOT SPAM       11        1358       99.2%

Average Accuracy: 98.16%
"""

import pickle
import sys
import time
import os
import re
import math

# Stopwords Reference: https://github.com/Alir3z4/stop-words/blob/0e438af98a88812ccc245cf31f93644709e70370/english.txt
STOPWORDS = {'all': 1, 'whys': 1, 'being': 1, 'over': 1, 'isnt': 1, 'through': 1, 'yourselves': 1, 'hell': 1, 'its': 1,
             'before': 1, 'wed': 1, 'with': 1, 'had': 1, 'should': 1, 'to': 1, 'lets': 1, 'under': 1, 'ours': 1,
             'has': 1, 'ought': 1, 'do': 1, 'them': 1, 'his': 1, 'very': 1, 'cannot': 1, 'they': 1, 'werent': 1,
             'not': 1, 'during': 1, 'yourself': 1, 'him': 1, 'nor': 1, 'wont': 1, 'did': 1, 'theyre': 1, 'this': 1,
             'she': 1, 'each': 1, 'havent': 1, 'where': 1, 'shed': 1, 'because': 1, 'doing': 1, 'theirs': 1, 'some': 1,
             'whens': 1, 'up': 1, 'are': 1, 'further': 1, 'ourselves': 1, 'out': 1, 'what': 1, 'for': 1, 'heres': 1,
             'while': 1, 'does': 1, 'above': 1, 'between': 1, 'youll': 1, 'be': 1, 'we': 1, 'who': 1, 'were': 1,
             'here': 1, 'hers': 1, 'by': 1, 'both': 1, 'about': 1, 'would': 1, 'wouldnt': 1, 'didnt': 1, 'ill': 1,
             'against': 1, 'arent': 1, 'youve': 1, 'theres': 1, 'or': 1, 'thats': 1, 'weve': 1, 'own': 1, 'whats': 1,
             'dont': 1, 'into': 1, 'youd': 1, 'whom': 1, 'down': 1, 'doesnt': 1, 'theyd': 1, 'couldnt': 1, 'your': 1,
             'from': 1, 'her': 1, 'hes': 1, 'there': 1, 'only': 1, 'been': 1, 'whos': 1, 'hed': 1, 'few': 1, 'too': 1,
             'themselves': 1, 'was': 1, 'until': 1, 'more': 1, 'himself': 1, 'on': 1, 'but': 1, 'you': 1, 'hadnt': 1,
             'shant': 1, 'mustnt': 1, 'herself': 1, 'than': 1, 'those': 1, 'he': 1, 'me': 1, 'myself': 1, 'theyve': 1,
             'these': 1, 'cant': 1, 'below': 1, 'of': 1, 'my': 1, 'could': 1, 'shes': 1, 'and': 1, 'ive': 1, 'then': 1,
             'wasnt': 1, 'is': 1, 'am': 1, 'it': 1, 'an': 1, 'as': 1, 'itself': 1, 'im': 1, 'at': 1, 'have': 1, 'in': 1,
             'any': 1, 'if': 1, 'again': 1, 'hasnt': 1, 'theyll': 1, 'no': 1, 'that': 1, 'when': 1, 'same': 1, 'id': 1,
             'how': 1, 'other': 1, 'which': 1, 'shell': 1, 'shouldnt': 1, 'our': 1, 'after': 1, 'most': 1, 'such': 1,
             'why': 1, 'wheres': 1, 'a': 1, 'hows': 1, 'off': 1, 'i': 1, 'youre': 1, 'well': 1, 'yours': 1, 'their': 1,
             'so': 1, 'the': 1, 'having': 1, 'once': 1}


class DecisionTreeNode:
    # Decision Tree node
    def __init__(self):
        pass

    left = None
    right = None
    word = ""


class FilesData:
    # Files data stored in this class
    # Variables to store number of times a word appeared in all the spam/non spam files
    def __init__(self):
        pass

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


class ProbabilityTable:
    # Probability Table  {word: [p being spam considering freq, p being not spam considering freq, p being spam
    # considering how many spam files have this word, p being not spam considering how many spam files have this word]}
    def __init__(self):
        pass

    probability_table = {}


def get_file_list(x_data_set):
    # Function to get list of files
    spam_file_list = os.listdir(x_data_set + "/spam")
    non_spam_file_list = os.listdir(x_data_set + "/notspam")
    return {"spam": spam_file_list, "notspam": non_spam_file_list}


def get_file_data(x_all_emails_list, x_data_set):
    # Function to get file data and store it in the FileData object
    spam_files = [{}, 0]
    not_spam_files = [{}, 0]

    fd.spam_files_count = len(x_all_emails_list["spam"])
    fd.n_spam_files_count = len(x_all_emails_list["notspam"])
    for file_name in x_all_emails_list["spam"]:
        spam_files = read_file(x_data_set + "/spam/" + file_name, spam_files, "spam")

    fd.total_words_spam_files = spam_files[1]
    fd.spam_words_f = spam_files[0]

    for file_name in x_all_emails_list["notspam"]:
        not_spam_files = read_file(x_data_set + "/notspam/" + file_name, not_spam_files, "notspam")

    fd.total_words_n_spam_files = not_spam_files[1]
    fd.n_spam_words_f = not_spam_files[0]

    return {"spam": spam_files, "notspam": not_spam_files}


def replace_line(line):
    line = line.lower()
    line = line.replace("return-path", " ")
    line = line.replace("received", " ")
    line = line.replace("date", " ")
    line = line.replace("subject", " ")
    line = line.replace("to", " ")
    line = line.replace("from", " ")
    line = line.replace("reply-to", " ")
    line = line.replace("message-id", " ")
    line = line.replace("mailto", " ")
    line = line.replace("sender", " ")
    line = line.replace("!--", " ")
    line = line.replace("--", " ")
    line = line.replace("=", " ")
    line = line.replace("!", " ")
    line = line.replace("[", " ")
    line = line.replace("]", " ")
    line = line.replace(", ", " ")
    line = line.replace(",", "")
    line = line.replace(";", " ")
    line = line.replace("(", " ")
    line = line.replace(")", " ")
    line = line.replace("<", " ")
    line = line.replace(">", " ")
    line = line.replace("\"", " ")
    line = line.replace(". ", " ")
    line = line.replace(": ", " ")
    line = line.replace("#", " ")
    if technique == "bayes":
        line = line.replace(".", " ")
    return line


# Regex for amount Reference: http://stackoverflow.com/questions/2150205/can-somebody-explain-a-money-regex-that-just-
# checks-if-the-value-matches-some-pa
# Regex for email http://stackoverflow.com/questions/8022530/python-check-for-valid-email-address
def read_file(file_name, current_data, file_type):
    words_in_this_file = {}
    words = current_data[0]
    total_no_of_words = 0
    file_data = open(file_name, 'r')

    for line in file_data:

        line = line.lower()
        line = replace_line(line)
        for word in line.split():
            actual_word = word

            if word in STOPWORDS:
                continue

            if word.isdigit():
                word = "numbernumber"

            if not word.isalpha() and word.find("@") == -1 and word.find(".com") == -1 and word.find(".net") == -1:
                continue

            if word.find("$") >= 0 or word.find("USD") >= 0:
                word = "dollardollar"

            word = re.sub('[^a-zA-Z0-9]', '', word)

            if word == '':
                continue

            total_no_of_words += 1

            if not (word in fd.words):
                fd.words[word] = 1
                fd.word_list_with_actual_word[word] = actual_word
            else:
                fd.words[word] += 1
            if word in words:
                words[word] += 1
            else:
                words[word] = 1

            if word in words_in_this_file:
                words_in_this_file[word] += 1
            else:
                if file_type == "spam":
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

    if file_type == "spam":
        fd.spam_files.append(words_in_this_file)
    else:
        fd.n_spam_files.append(words_in_this_file)
    return [words, current_data[1] + total_no_of_words]


def get_p_distribution():
    # Function to calculate probability distribution
    p_if_not_present_f = 1.0 / fd.total_words_spam_files
    p_if_not_present_01 = 1.0 / fd.spam_files_count
    for word in fd.words:
        pt.probability_table[word] = [p_if_not_present_f, p_if_not_present_f, p_if_not_present_01, p_if_not_present_01]
        if word in fd.no_of_files_with_this_word_in_spam:
            pt.probability_table[word][0] = (1.0 * fd.spam_words_f[word]) / fd.total_words_spam_files
            pt.probability_table[word][2] = (1.0 * fd.no_of_files_with_this_word_in_spam[word]) / fd.spam_files_count

        if word in fd.no_of_files_with_this_word_in_n_spam:
            pt.probability_table[word][1] = (1.0 * fd.n_spam_words_f[word]) / fd.total_words_n_spam_files
            pt.probability_table[word][3] = (
                                                1.0 * fd.no_of_files_with_this_word_in_n_spam[
                                                    word]) / fd.n_spam_files_count
    return 0


def write_model_bayes(file_path):
    output_file = open(file_path, 'w')
    for word in pt.probability_table:
        output_file.write('%s %s ' % (word, pt.probability_table[word][0]))
        output_file.write('%s %s ' % (pt.probability_table[word][1], pt.probability_table[word][2]))
        output_file.write('%s\n' % (pt.probability_table[word][3]))
    output_file.close()


def read_model(file_path):
    input_file = open(file_path, 'r')
    for line in input_file:
        words = line.split()
        pt.probability_table[words[0]] = [float(words[1]), float(words[2]), float(words[3]), float(words[4])]
    input_file.close()


def find_p_given_words(x_files):
    result = [0, 0, 0, 0]
    for mail in x_files:
        probability_01 = 1.0
        probability_f = 1.0
        for word in mail:
            if word in pt.probability_table:
                for i in range(mail[word]):
                    probability_f *= pt.probability_table[word][0] / pt.probability_table[word][1]
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


def find_most_probable_words(x, y):
    words = [""] * 20
    probability = [0.0] * 20
    for word in pt.probability_table:
        p_of_word = pt.probability_table[word][x] / pt.probability_table[word][y]
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


def print_most_probable_words(words):
    result = []
    for word in words:
        result.append(fd.word_list_with_actual_word[word])
    return result


def train_bayes():
    print "Reading files....."
    email_directories = get_file_list(data_set)
    get_file_data(email_directories, data_set)
    print "Files read"
    print "Training on bayes net....."
    get_p_distribution()
    print "Training completed"
    print "Writing model to file....."
    write_model_bayes(model_file)
    print "Model saved to file"
    print ""
    print "Top 10 words::"

    print "\n\ttaking words as binary:"
    g_words = find_most_probable_words(2, 3)
    print "\t\tfor P(S=1|w):" + str(print_most_probable_words(g_words))

    g_words = find_most_probable_words(3, 2)
    print "\n\t\tfor P(S=0|w):" + str(print_most_probable_words(g_words))

    g_words = find_most_probable_words(0, 1)
    print "\n\n\ttaking frequency in account:"
    print "\n\t\tfor P(S=1|w) :" + str(print_most_probable_words(g_words))

    g_words = find_most_probable_words(1, 0)
    print "\n\t\tfor P(S=0|w):" + str(print_most_probable_words(g_words))

    print "\n\nCompleted"


def test_bayes():
    print "Reading model....."
    read_model(model_file)
    print "Model read"
    print "Reading files....."
    email_directories = get_file_list(data_set)
    get_file_data(email_directories, data_set)
    print "Files read"
    print "Finding results....."

    result1 = find_p_given_words(fd.spam_files)
    result2 = find_p_given_words(fd.n_spam_files)

    confusion_matrix_f = [result1[0], result1[1], result2[0], result2[1]]
    confusion_matrix_01 = [result1[2], result1[3], result2[2], result2[3]]

    print "\nConfusion Matrix \n(Model 1: Words as binary features):"
    print_confusion_matrix(confusion_matrix_01)

    print "\n\nConfusion Matrix \n(Model 2: Words frequency taken into account):"
    print_confusion_matrix(confusion_matrix_f)
    print "\nTesting completed"


def find_word_based_on_entropy(table, words, word_for_s):
    s_ns_count_for_word = {}  # word :[word present and mail counted as spam, word present and mail counted as spam,
    # word not present and mail counted as spam, word not present and mail counted as spam,]

    min_avg_disorder = 10
    min_avg_disorder_word = ""
    left_tree_p_spam = 0.0
    right_tree_p_spam = 0.0
    for word in words:
        s_ns_count_for_word[word] = [0, 0, 0, 0]           # Considering f

    for entry in table:
        for word in words:
            if word in entry:
                if word_for_s in entry:
                    s_ns_count_for_word[word][0] += 1
                else:
                    s_ns_count_for_word[word][1] += 1
            else:
                if word_for_s in entry:
                    s_ns_count_for_word[word][2] += 1
                else:
                    s_ns_count_for_word[word][3] += 1

    for word in words:
        [avg_disorder, m1_by_m_r_branch, m1_by_m_l_branch] = find_disorder(s_ns_count_for_word[word])
        if avg_disorder < min_avg_disorder:
            right_tree_p_spam = m1_by_m_r_branch
            left_tree_p_spam = m1_by_m_l_branch
            min_avg_disorder = avg_disorder
            min_avg_disorder_word = word
    return [min_avg_disorder_word, left_tree_p_spam, right_tree_p_spam]


def find_word_based_on_entropy_f(table, words, word_for_s):
    s_ns_count_for_word = {}  # word :[word present and mail counted as spam, word present and mail counted as spam,
    # word not present and mail counted as spam, word not present and mail counted as spam,]
    min_avg_disorder = 20000000
    min_avg_disorder_word = ""
    left_tree_p_spam = 0.0
    right_tree_p_spam = 0.0
    for word in words:
        s_ns_count_for_word[word] = [0, 0, 0, 0]           # Considering f

    for entry in table:
        for word in words:
            if word in entry:
                if word_for_s in entry:
                    s_ns_count_for_word[word][0] += entry[word]
                else:
                    s_ns_count_for_word[word][1] += entry[word]
            else:
                if word_for_s in entry:
                    s_ns_count_for_word[word][2] += 1
                else:
                    s_ns_count_for_word[word][3] += 1

    for word in words:
        [avg_disorder, m1_by_m_r_branch, m1_by_m_l_branch] = find_disorder(s_ns_count_for_word[word])
        if avg_disorder < min_avg_disorder:
            right_tree_p_spam = m1_by_m_r_branch
            left_tree_p_spam = m1_by_m_l_branch
            min_avg_disorder = avg_disorder
            min_avg_disorder_word = word
    return [min_avg_disorder_word, left_tree_p_spam, right_tree_p_spam]


def find_disorder(count_list):
    total_samples_branch_true = count_list[0] + count_list[1]
    total_samples_branch_false = count_list[2] + count_list[3]
    total_samples = total_samples_branch_true + total_samples_branch_false

    if total_samples_branch_true == 0:
        total_samples_branch_true = 1
    if total_samples_branch_false == 0:
        total_samples_branch_false = 1

    m1_by_m_r_branch = 1.0 * count_list[0] / total_samples_branch_true
    m2_by_m_r_branch = 1.0 * count_list[1] / total_samples_branch_true

    # Branch true: word present
    if m1_by_m_r_branch == 0 or m2_by_m_r_branch == 0:
        entropy_branch_true = 0
    else:
        entropy_branch_true = -1.0 * (m1_by_m_r_branch * math.log(m1_by_m_r_branch, 2) + m2_by_m_r_branch *
                                      math.log(m2_by_m_r_branch, 2))

    # Branch false: word not present
    m1_by_m_l_branch = 1.0 * count_list[2] / total_samples_branch_false
    m2_by_m_l_branch = 1.0 * count_list[3] / total_samples_branch_false
    if m1_by_m_l_branch == 0 or m2_by_m_l_branch == 0:
        entropy_branch_false = 0
    else:
        entropy_branch_false = -1.0 * (m1_by_m_l_branch * math.log(m1_by_m_l_branch, 2) + m2_by_m_l_branch *
                                       math.log(m2_by_m_l_branch, 2))

    total_samples_branch_true = count_list[0] + count_list[1]
    total_samples_branch_false = count_list[2] + count_list[3]
    avg_disorder = (1.0 * total_samples_branch_true / total_samples) * entropy_branch_true + \
                   (1.0 * total_samples_branch_false / total_samples) * entropy_branch_false
    return [avg_disorder, m1_by_m_r_branch, m1_by_m_l_branch]


def split_table(word, table):
    split_right = []
    split_left = []
    for entry in table:
        if word in entry:
            split_right.append(entry)
        else:
            split_left.append(entry)

    return [split_left, split_right]


def main_generate_decision_tree(word_list, table, word_for_s, depth):
    dt_01 = rec_create_decision_tree(word_list, table, word_for_s, depth, True)
    dt_f = rec_create_decision_tree(word_list, table, word_for_s, depth, False)
    return [dt_01, dt_f]


def rec_create_decision_tree(word_list, table, word_for_s, depth, type_01):
    max_depth = 30
    node = DecisionTreeNode()
    if len(word_list) != 0 and len(table) != 0:
        if type_01:
            [word, p_spam_l_branch, p_spam_r_branch] = find_word_based_on_entropy(table, word_list, word_for_s)
        else:
            [word, p_spam_l_branch, p_spam_r_branch] = find_word_based_on_entropy_f(table, word_list, word_for_s)
        [left_split_table, right_split_table] = split_table(word, table)
        word_list.remove(word)

        #####
        # Create tree here
        #####
        node.word = word
        if len(word_list) == 0 or depth > max_depth:
            if p_spam_l_branch > 0.5:
                # Left branch Spam
                node.left = word_for_spam
            else:
                node.left = word_for_n_spam
            if p_spam_r_branch > 0.5:
                # Left branch Spam
                node.right = word_for_spam
            else:
                node.right = word_for_n_spam
            return node

        if True:
            # Left node
            if p_spam_l_branch >= 0.9:
                # Left branch Spam
                node.left = word_for_spam
            elif p_spam_l_branch <= 0.1:
                # Left branch not spam
                node.left = word_for_n_spam
            else:
                # Make subtree
                if len(left_split_table) == 0:
                    if p_spam_l_branch > 0.5:
                        node.left = word_for_spam
                    else:
                        node.right = word_for_n_spam
                else:
                    node.left = rec_create_decision_tree(word_list, left_split_table, word_for_s, depth + 1, type_01)
                # result = [rec_generate_decision_tree(word_list, left_split_table, word_for_s)[0]]

            # Right node
            if p_spam_r_branch >= 0.9:
                # Right branch Spam
                node.right = word_for_spam
            elif p_spam_r_branch <= 0.1:
                # Right branch not spam
                node.right = word_for_n_spam
            else:
                # Make subtree
                if len(right_split_table) == 0:
                    if p_spam_r_branch > 0.5:
                        node.right = word_for_spam
                    else:
                        node.right = word_for_n_spam
                else:
                    node.right = rec_create_decision_tree(word_list, right_split_table, word_for_s, depth + 1, type_01)
                # result.append(rec_generate_decision_tree(word_list, right_split_table, word_for_s)[0])
        return node
    print "Error"
    return


def test_file_dt(my_file, dt):
    while dt != word_for_spam and dt != word_for_n_spam:
        if dt.word in my_file:
            dt = dt.right
        else:
            dt = dt.left

    if dt == word_for_spam:
        return True
    else:
        return False


def train_dt():
    print "Reading files....."
    email_directories = get_file_list(data_set)
    get_file_data(email_directories, data_set)

    all_files = []
    for x_file in fd.spam_files:
        x_file[word_for_spam] = 1
        all_files.append(x_file)
    for x_file in fd.n_spam_files:
        all_files.append(x_file)
    print "Files read"
    print "Creating decision tree....."
    my_words = []
    for this_word in fd.words:
        if fd.words[this_word] > 10:
            my_words.append(this_word)
    return main_generate_decision_tree(my_words, all_files, word_for_spam, 0)


def test_dt_head(head):
    confusion_matrix = [0, 0, 0, 0]
    for x_file in fd.spam_files:
        if test_file_dt(x_file, head):
            confusion_matrix[0] += 1
        else:
            confusion_matrix[1] += 1
    for x_file in fd.n_spam_files:
        if test_file_dt(x_file, head):
            confusion_matrix[2] += 1
        else:
            confusion_matrix[3] += 1
    return confusion_matrix


def test_dt():
    print "Reading Model....."
    [dt_01, dt_f] = read_model_dt(model_file)
    print "Read Model"
    print "Reading files....."
    email_directories = get_file_list(data_set)
    get_file_data(email_directories, data_set)
    print "Files read"
    print "Finding results....."
    confusion_matrix_01 = test_dt_head(dt_01)
    confusion_matrix_f = test_dt_head(dt_f)
    print "\nConfusion Matrix \n(Model 1: Words as binary features):"
    print_confusion_matrix(confusion_matrix_01)

    print "\nConfusion Matrix \n(Model 2: Words as frequency):"
    print_confusion_matrix(confusion_matrix_f)


def write_model_dt(file_path, x_dts):
    [dt_01, dt_f] = x_dts
    output_file = open(file_path+".pkl", 'wb')
    pickle.dump(dt_01, output_file, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dt_f, output_file, pickle.HIGHEST_PROTOCOL)
    output_file.close()


def read_model_dt(file_path):
    input_file = open(file_path+".pkl", 'rb')
    dt_01 = pickle.load(input_file)
    dt_f = pickle.load(input_file)

    input_file.close()
    return [dt_01, dt_f]


def reformat_size_4(my_str):
    while len(my_str) < 5:
        my_str = " " + my_str
    return my_str


def print_confusion_matrix(confusion_matrix):
    cm_00 = reformat_size_4(str(confusion_matrix[0]))
    cm_01 = reformat_size_4(str(confusion_matrix[1]))
    cm_10 = reformat_size_4(str(confusion_matrix[2]))
    cm_11 = reformat_size_4(str(confusion_matrix[3]))
    accuracy_s = round(confusion_matrix[0] * 100.0/(confusion_matrix[0] + confusion_matrix[1]), 2)
    accuracy_ns = round(confusion_matrix[3] * 100.0 / (confusion_matrix[2] + confusion_matrix[3]), 2)
    avg = round((accuracy_ns + accuracy_s)/2, 2)
    a_s = reformat_size_4(str(accuracy_s) + "%")
    a_ns = reformat_size_4(str(accuracy_ns) + "%")

    print '%-12s%-12s%-12s%-12s' % (" ", "SPAM ", "NOT SPAM ", "ACCURACY ")
    print '%-12s%-12s%-12s%-12s' % ("SPAM ", cm_00 + " ", cm_01 + " ", a_s + " ")
    print '%-12s%-12s%-12s%-12s' % ("NOT SPAM ", cm_10 + " ", cm_11 + " ", a_ns + " ")
    print "\nAverage Accuracy: " + str(avg) + "%"


def print_decision_model(dt, height):
    print "\tNode:" + dt.word
    call_left = False
    call_right = False
    if dt.left == word_for_spam:
        print "\t\tLeft Node(Leaf): Spam"
    elif dt.left == word_for_n_spam:
        print "\t\tLeft Node(Leaf): Not Spam"
    elif dt.left != word_for_spam and dt.left != word_for_n_spam:
        print "\t\tLeft Node:" + dt.left.word
        call_left = True

    if dt.right == word_for_spam:
        print "\t\tRight Node(Leaf): Spam"
    elif dt.right == word_for_n_spam:
        print "\t\tRight Node(Leaf): Not Spam"
    elif dt.right != word_for_spam and dt.right != word_for_n_spam:
        print "\t\tRight Node:" + dt.right.word
        call_right = True

    if call_left and height < 4:
        print_decision_model(dt.left, height + 1)

    if call_right and height < 4:
        print_decision_model(dt.right, height + 1)


start_time = time.time()
print time.asctime(time.localtime(time.time()))
print ""
###################################


(mode, technique, data_set, model_file) = sys.argv[1:]

fd = FilesData()
pt = ProbabilityTable()
word_for_spam = "SSSSpam"
word_for_n_spam = "NotSSSSpam"

if mode == "train" and technique == "bayes":
    train_bayes()

if mode == "test" and technique == "bayes":
    test_bayes()

if mode == "train" and technique == "dt":
    dts = train_dt()
    print "Created decision tree"
    print "Writing model to file....."
    write_model_dt(model_file, dts)
    print "Model saved to file"
    print "Training completed"
    print "\nDECISION TREE: Model 1: word as binary feature"
    print_decision_model(dts[0], 0)
    print "\nDECISION TREE: Model 2: word frequency taken into account"
    print_decision_model(dts[1], 0)

if mode == "test" and technique == "dt":
    test_dt()
    print "\nTesting completed"

#######################
end_time = time.time()
print time.asctime(time.localtime(time.time()))
print end_time - start_time