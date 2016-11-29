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
    email_directories = get_file_list(data_set)
    get_file_data(email_directories, data_set)
    get_p_distribution()
    write_model_bayes(model_file)

    print ""
    g_words = find_most_probable_words(0, 1)
    print "P(S=1|w) considering f:" + str(print_most_probable_words(g_words))
    g_words = find_most_probable_words(2, 3)
    print "P(S=1|w) considering p:" + str(print_most_probable_words(g_words))
    g_words = find_most_probable_words(1, 0)
    print "P(S=0|w) considering f:" + str(print_most_probable_words(g_words))
    g_words = find_most_probable_words(3, 2)
    print "P(S=0|w) considering p:" + str(print_most_probable_words(g_words))


def test_bayes():
    read_model(model_file)
    email_directories = get_file_list(data_set)
    get_file_data(email_directories, data_set)
    result1 = find_p_given_words(fd.spam_files)
    result2 = find_p_given_words(fd.n_spam_files)
    print "ConfusionMatrix where index 0 = Spam and index 1 = Not Spam, row i column j show the number "
    print "of test exemplars whose correct label is i, but that were classified as j"

    confusion_matrix_f = [[result1[0], result1[1]], [result2[0], result2[1]]]
    confusion_matrix_01 = [[result1[2], result1[3]], [result2[2], result2[3]]]

    print "freq. of word: " + str(confusion_matrix_f)
    print "pres. of word: " + str(confusion_matrix_01)


def find_word_based_on_entropy(table, words, word_for_s):
    s_ns_count_for_word = {}  # word :[word present and mail counted as spam, word present and mail counted as spam,
    # word not present and mail counted as spam, word not present and mail counted as spam,]
    min_avg_disorder = 10
    min_avg_disorder_word = ""
    left_tree_p_spam = 0.0
    right_tree_p_spam = 0.0
    for word in words:
        s_ns_count_for_word[word] = [0, 0, 0, 0]

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
        total_samples_branch_true = s_ns_count_for_word[word][0] + s_ns_count_for_word[word][1]
        total_samples_branch_false = s_ns_count_for_word[word][2] + s_ns_count_for_word[word][3]
        total_samples = total_samples_branch_true + total_samples_branch_false

        if total_samples_branch_true == 0:
            total_samples_branch_true = 1
        if total_samples_branch_false == 0:
            total_samples_branch_false = 1

        # Branch true:word present
        m1_by_m_r_branch = 1.0 * s_ns_count_for_word[word][0] / total_samples_branch_true
        m2_by_m_r_branch = 1.0 * s_ns_count_for_word[word][1] / total_samples_branch_true
        if m1_by_m_r_branch == 0 or m2_by_m_r_branch == 0:
            entropy_branch_true = 0
        else:
            entropy_branch_true = -1.0 * (m1_by_m_r_branch * math.log(m1_by_m_r_branch, 2) + m2_by_m_r_branch *
                                          math.log(m2_by_m_r_branch, 2))

        # Branch true:word present
        m1_by_m_l_branch = 1.0 * s_ns_count_for_word[word][2] / total_samples_branch_false
        m2_by_m_l_branch = 1.0 * s_ns_count_for_word[word][3] / total_samples_branch_false
        if m1_by_m_l_branch == 0 or m2_by_m_l_branch == 0:
            entropy_branch_false = 0
        else:
            entropy_branch_false = -1.0 * (m1_by_m_l_branch * math.log(m1_by_m_l_branch, 2) + m2_by_m_l_branch *
                                           math.log(m2_by_m_l_branch, 2))

        total_samples_branch_true = s_ns_count_for_word[word][0] + s_ns_count_for_word[word][1]
        total_samples_branch_false = s_ns_count_for_word[word][2] + s_ns_count_for_word[word][3]
        avg_disorder = (1.0 * total_samples_branch_true / total_samples) * entropy_branch_true + \
                       (1.0 * total_samples_branch_false / total_samples) * entropy_branch_false

        if avg_disorder < min_avg_disorder:
            right_tree_p_spam = m1_by_m_r_branch
            left_tree_p_spam = m1_by_m_l_branch
            min_avg_disorder = avg_disorder
            min_avg_disorder_word = word
    return [min_avg_disorder_word, left_tree_p_spam, right_tree_p_spam]


def split_table(word, table):
    split_right = []
    split_left = []
    for entry in table:
        if word in entry:
            split_right.append(entry)
        else:
            split_left.append(entry)

    return [split_left, split_right]


def rec_generate_decision_tree(word_list, table, word_for_s):
    node = DecisionTreeNode()
    result = []
    while len(word_list) != 0 and len(table) != 0:
        [word, p_spam_l_branch, p_spam_r_branch] = find_word_based_on_entropy(table, word_list, word_for_s)
        [left_split_table, right_split_table] = split_table(word, table)
        word_list.remove(word)

        #####
        # Create tree here
        #####
        node.word = word
        if len(word_list) == 0 or len(left_split_table) == 0:
            if p_spam_l_branch > 0.5:
                # Left branch Spam
                node.left = word_for_spam
                result = ["Spam"]
            else:
                node.left = word_for_n_spam
                result = ["NotSpam"]

        if len(word_list) == 0 or len(right_split_table) == 0:
            if p_spam_r_branch > 0.5:
                # Left branch Spam
                node.right = word_for_spam
                result.append("Spam")
            else:
                node.right = word_for_n_spam
                result.append("NotSpam")
        else:
            # Left node
            if p_spam_l_branch >= 0.9:
                # Left branch Spam
                node.left = word_for_spam
                result = ["Spam"]
            elif p_spam_l_branch <= 0.1:
                # Left branch not spam
                node.left = word_for_n_spam
                result = ["NotSpam"]
            else:
                # Make subtree
                node.left = rec_generate_decision_tree(word_list, left_split_table, word_for_s)
                # result = [rec_generate_decision_tree(word_list, left_split_table, word_for_s)[0]]

            # Right node
            if p_spam_r_branch >= 0.9:
                # Right branch Spam
                node.right = word_for_spam
                result.append("Spam")
            elif p_spam_r_branch <= 0.1:
                # Right branch not spam
                node.right = word_for_n_spam
                result.append("NotSpam")
            else:
                # Make subtree
                node.right = rec_generate_decision_tree(word_list, left_split_table, word_for_s)
                # result.append(rec_generate_decision_tree(word_list, right_split_table, word_for_s)[0])

    return node


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
    email_directories = get_file_list(data_set)
    get_file_data(email_directories, data_set)
    all_files = []
    for x_file in fd.spam_files:
        x_file[word_for_spam] = 1
        all_files.append(x_file)
    for x_file in fd.n_spam_files:
        all_files.append(x_file)
    my_words = []
    for this_word in fd.words:
        if fd.words[this_word] > 10:
            if this_word not in STOPWORDS:
                my_words.append(this_word)
    return rec_generate_decision_tree(my_words, all_files, word_for_spam)


def test_dt():
    head = read_model_dt(model_file)
    email_directories = get_file_list(data_set)
    get_file_data(email_directories, data_set)
    confusion_matrix_01 = [0, 0, 0, 0]
    for x_file in fd.spam_files:
        if test_file_dt(x_file, head):
            confusion_matrix_01[0] += 1
        else:
            confusion_matrix_01[1] += 1
    for x_file in fd.n_spam_files:
        if test_file_dt(x_file, head):
            confusion_matrix_01[2] += 1
        else:
            confusion_matrix_01[3] += 1
    print confusion_matrix_01


def write_model_dt(file_path, head):
    output_file = open(file_path+".pkl", 'wb')
    pickle.dump(head, output_file, pickle.HIGHEST_PROTOCOL)
    output_file.close()


def read_model_dt(file_path):
    input_file = open(file_path+".pkl", 'rb')
    dt = pickle.load(input_file)
    input_file.close()
    return dt

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
    dt_head = train_dt()
    write_model_dt(model_file, dt_head)

if mode == "test" and technique == "dt":
    test_dt()


#######################
end_time = time.time()
print ""
print time.asctime(time.localtime(time.time()))
print end_time - start_time
