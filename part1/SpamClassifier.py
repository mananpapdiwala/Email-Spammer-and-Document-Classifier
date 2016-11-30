import pickle
import sys
import time
import os
import re
import math


STOPWORDS = {'all', 'pointing', 'whoever', 'four', 'go', 'mill', 'oldest', 'seemed', 'whose', 'certainly', 'young', 'p',
             'presents', 'to', 'asking', 'those', 'under', 'far', 'every', 'yourselves', 'presented', 'did', 'turns',
             'large', 'small', 'havent', 'thereupon', 'parted', 'smaller', 'says', 'ten', 'yourself', 'whens', 'here',
             'second', 'further', 'even', 'what', 'heres', 'wouldnt', 'anywhere', 'above', 'new', 'ever', 'thin', 'men',
             'full', 'mustnt', 'youd', 'sincere', 'youngest', 'let', 'groups', 'others', 'alone', 'having', 'almost',
             'along', 'fifteen', 'great', 'didnt', 'k', 'wherever', 'amount', 'arent', 'thats', 'via', 'besides', 'put',
             'everybody', 'from', 'working', 'two', 'next', 'call', 'therefore', 'taken', 'themselves', 'use', 'evenly',
             'thru', 'until', 'today', 'more', 'knows', 'clearly', 'becomes', 'hereby', 'herein', 'downing', 'hereupon',
             'everywhere', 'known', 'cases', 'must', 'me', 'states', 'none', 'room', 'f', 'this', 'work', 'itself', 'l',
             'nine', 'can', 'mr', 'making', 'my', 'numbers', 'give', 'high', 'weve', 'something', 'want', 'needs', 'eg',
             'end', 'turn', 'rather', 'meanwhile', 'how', 'itse', 'shouldnt', 'y', 'may', 'after', 'them', 'whenever',
             'such', 'man', 'a', 'third', 'q', 'so', 'keeps', 'order', 'six', 'furthering', 'indeed', 'over', 'move',
             'years', 'ended', 'isnt', 'through', 'fify', 'hell', 'still', 'its', 'before', 'beside', 'group', 'thence',
             'somewhere', 'interesting', 'better', 'differently', 'ours', 'might', 'then', 'non', 'good', 'somebody',
             'greater', 'thereby', 'eleven', 'downs', 'they', 'not', 'now', 'nor', 'wont', 'gets', 'hereafter',
             'always', 'whither', 'doesnt', 'each', 'found', 'went', 'side', 'everyone', 'doing',
             'year', 'our', 'beyond', 'out', 'opened', 'since', 'forty', 're', 'got', 'myse', 'shows', 'turning',
             'differ', 'quite', 'whereupon', 'members', 'ask', 'anyhow', 'wanted', 'g', 'could', 'needing', 'keep',
             'thing', 'place', 'w', 'ltd', 'hence', 'onto', 'think', 'first', 'already', 'seeming', 'thereafter',
             'number', 'one', 'done', 'another', 'thick', 'open', 'given', 'needed', 'ordering', 'twenty', 'top',
             'system', 'least', 'name', 'anyone', 'their', 'too', 'hundred', 'gives', 'interests', 'shell', 'mostly',
             'behind', 'nobody', 'took', 'part', 'hadnt', 'herself', 'than', 'kind', 'b', 'showed', 'older', 'likely',
             'nevertheless', 'r', 'were', 'toward', 'and', 'sees', 'wasnt', 'turned', 'few', 'say', 'have', 'need',
             'seem', 'saw', 'orders', 'latter', 'that', 'also', 'take', 'which', 'wanting', 'sure', 'shall', 'knew',
             'wells', 'most', 'eight', 'amongst', 'nothing', 'why', 'parting', 'noone', 'later', 'm', 'amoungst', 'mrs',
             'points', 'fact', 'show', 'anyway', 'ending', 'find', 'state', 'should', 'only', 'going', 'pointed', 'do',
             'his', 'get', 'de', 'cannot', 'longest', 'werent', 'during', 'him', 'areas', 'h', 'cry', 'she', 'x',
             'where', 'theirs', 'we', 'whys', 'see', 'computer', 'are', 'best', 'said', 'ways', 'away', 'please',
             'enough', 'smallest', 'between', 'neither', 'youll', 'across', 'ends', 'never', 'opening', 'however',
             'come', 'both', 'c', 'last', 'many', 'ill', 'whereafter', 'against', 'etc', 's', 'became', 'faces',
             'whole', 'asked', 'con', 'among', 'co', 'afterwards', 'point', 'seems', 'whatever', 'furthered', 'hers',
             'moreover', 'throughout', 'furthers', 'puts', 'three', 'been', 'whos', 'whom', 'much', 'dont', 'wherein',
             'interest', 'empty', 'wants', 'fire', 'beforehand', 'else', 'worked', 'an', 'former', 'present', 'case',
             'myself', 'theyve', 'these', 'bill', 'n', 'will', 'while', 'theres', 'ive', 'would', 'backing', 'is',
             'thus', 'it', 'cant', 'someone', 'im', 'in', 'ie', 'id', 'if', 'different', 'inc', 'perhaps', 'things',
             'make', 'same', 'any', 'member', 'parts', 'several', 'higher', 'used', 'upon', 'uses', 'thoughts', 'hows',
             'off', 'whereby', 'largely', 'i', 'youre', 'well', 'anybody', 'finds', 'thought', 'without', 'greatest',
             'very', 'the', 'otherwise', 'yours', 'latest', 'newest', 'just', 'less', 'being', 'when', 'detail',
             'front', 'rooms', 'facts', 'yet', 'wed', 'had', 'except', 'sometimes', 'lets', 'interested', 'has',
             'ought', 'gave', 'around', 'big', 'showing', 'possible', 'early', 'five', 'know', 'like', 'necessary', 'd',
             'herse', 'theyre', 'either', 'fully', 'become', 'works', 'grouping', 'therein', 'twelve', 'shed', 'once',
             'because', 'old', 'often', 'namely', 'some', 'back', 'towards', 'shes', 'mine', 'himse', 'thinks', 'for',
             'bottom', 'though', 'per', 'everything', 'does', 't', 'be', 'who', 'seconds', 'nowhere', 'although',
             'sixty', 'by', 'on', 'about', 'goods', 'asks', 'anything', 'of', 'o', 'whence', 'youve', 'or', 'own',
             'whats', 'formerly', 'into', 'within', 'due', 'down', 'hes', 'beings', 'right', 'theyd', 'couldnt', 'your',
             'her', 'area', 'downed', 'there', 'long', 'hed', 'way', 'was', 'opens', 'himself', 'elsewhere', 'becoming',
             'but', 'somehow', 'newer', 'shant', 'highest', 'with', 'he', 'made', 'places', 'whether', 'j', 'up', 'us',
             'below', 'un', 'problem', 'z', 'clear', 'v', 'ordered', 'certain', 'describe', 'am', 'general', 'as',
             'sometime', 'at', 'face', 'fill', 'again', 'hasnt', 'theyll', 'no', 'whereas', 'generally', 'backs',
             'ourselves', 'grouped', 'other', 'latterly', 'wheres', 'you', 'really', 'felt', 'problems', 'important',
             'sides', 'began', 'younger', 'e', 'longer', 'came', 'backed', 'together', 'u', 'presenting', 'serious'}


class DecisionTreeNode:
    def __init__(self):
        pass

    left = None
    right = None
    word = ""


class FilesData:
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

    def re_initialize(self):
        self.spam_words_f = {}
        self.n_spam_words_f = {}

        # Variables to store number probability of a word appearing in spam/non spam files considering freq. of word
        self.p_spam_words_f = {}
        self.p_n_spam_words_f = {}

        # Variable that store no. of files that has the particular word
        self.no_of_files_with_this_word_in_spam = {}
        self.no_of_files_with_this_word_in_n_spam = {}

        # Variable that store probability of word appearing in a document
        self.p_spam_words_01 = {}
        self.p_n_spam_words_01 = {}

        # Variables to store file count
        self.spam_files_count = 0
        self.n_spam_files_count = 0

        # Variables to store word count
        self.total_words_spam_files = 0
        self.total_words_n_spam_files = 0

        # Variables that store file as a word frequency list
        self.spam_files = []
        self.n_spam_files = []

        # List of words
        self.words = {}

        # Actual word for the word
        self.word_list_with_actual_word = {}


class ProbabilityTable:
    # Probability Table  {word: [p being spam considering freq, p being not spam considering freq, p being spam
    # considering how many spam files have this word, p being not spam considering how many spam files have this word]}
    def __init__(self):
        pass

    probability_table = {}


def get_file_list(x_data_set):
    spam_file_list = os.listdir(x_data_set + "/spam")
    non_spam_file_list = os.listdir(x_data_set + "/notspam")
    return {"spam": spam_file_list, "notspam": non_spam_file_list}


def get_file_data(x_all_emails_list, x_data_set):
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


# Regex for amount Reference: http://stackoverflow.com/questions/2150205/can-somebody-explain-a-money-regex-that-just-
# checks-if-the-value-matches-some-pa
# Regex for email http://stackoverflow.com/questions/8022530/python-check-for-valid-email-address
def read_file(file_name, current_data, file_type):
    words_in_this_file = {}
    words = current_data[0]
    total_no_of_words = 0
    file_data = open(file_name, 'r')

    for line in file_data:
        for word in line.split():
            actual_word = word

            if not word.find("$"):
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

    g_words = find_most_probable_words(0, 1)
    print "\n\ttaking frequency in account:"
    print "\n\t\tfor P(S=1|w) :" + str(print_most_probable_words(g_words))

    g_words = find_most_probable_words(1, 0)
    print "\n\t\tfor P(S=0|w):" + str(print_most_probable_words(g_words))

    print "\n\n\ttaking words as binary:"
    g_words = find_most_probable_words(2, 3)
    print "\t\tfor P(S=1|w):" + str(print_most_probable_words(g_words))

    g_words = find_most_probable_words(3, 2)
    print "\n\t\tP(S=0|w):" + str(print_most_probable_words(g_words))

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

    print "\nConfusion Matrix (Model 1: Words as binary features):"
    print_confusion_matrix(confusion_matrix_f)
    print "Confusion Matrix (Model 2: Words as freq):"
    print_confusion_matrix(confusion_matrix_01)
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
            if this_word not in STOPWORDS:
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
    print "\nConfusion Matrix"
    print "01:\n"
    print_confusion_matrix(confusion_matrix_01)
    print "f:\n"
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


def print_confusion_matrix(confusion_matrix):
    print "\t\t\t---------------------"
    print "\t\t\t| Spam\t| Not Spam \t|"
    print "\t\t\t|-------|----------\t|"
    print "Spam\t\t| " + str(confusion_matrix[0]) + "\t| " + str(confusion_matrix[1]) + "\t\t|"
    print "Not Spam\t| " + str(confusion_matrix[2]) + "\t| " + str(confusion_matrix[3]) + "\t\t|"
    print "\t\t\t---------------------"

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

if mode == "test" and technique == "dt":
    test_dt()
    print "\nTesting completed"

#######################
end_time = time.time()
print ""
print time.asctime(time.localtime(time.time()))
print end_time - start_time
