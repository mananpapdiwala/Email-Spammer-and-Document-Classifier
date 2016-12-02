# Usage:
#   ./topics mode dataset-directory model-file [fraction]
# Accuracy for fraction = 0.0 : 3.30589484865
# Accuracy for fraction = 0.1 : 74.2166755178
# Accuracy for fraction = 0.2 : 79.8990971853
# Accuracy for fraction = 0.3 : 80.1779075943
# Accuracy for fraction = 0.4 : 80.7620817844
# Accuracy for fraction = 0.5 : 82.1295804567
# Accuracy for fraction = 0.6 : 83.297928837
# Accuracy for fraction = 0.7 : 82.7403080191
# Accuracy for fraction = 0.8 : 83.4838024429
# Accuracy for fraction = 0.9 : 83.4439723845
# Accuracy for fraction = 1.0 : 83.8422729687
#

# Some assumptions:
# There are two folders train and test in the same directory that contains this code file.
# There are 20 folders each in both train and test folder having the folder name as the topic name.
# The folder and file  starting with . are ignored

# Along with the code file we are also uplodaing a model.txt and distinctive_words.txt we have created using fration 1.0 and the train data you provided

# Fully Supervised Learning:
# We are reading all the files and maintaining a dictionary called p_w_t
# It's keys are all the different words found during training
# The value of each word key is another dicitonary which contains distribution of that word in different classification
# In short we are maintaining the count for each word in different classification
# The probability is then calculated by dividing the total word count in a topic

# The test part then uses this probability to find the probability of a test word for all topics
# The topic that gives maximum probability is used to labelled the test word.

# Semi Supervised Learning:
# We multiply the fraction by 100
# While reading each file we generate a random number between 1 and 100
# If that number is greater than fraction multipled by 100 we label the document as unlabelled or else we use the supervised learning on that file
# All the unknown files are then used as test files and the trained data based on partial files is used to label these unknown files and create the final trained model

# The actual test files are then labelled using this trained model

# Unsupervised Learning:
# Each word is randomly assigned label
# This process is run within a for loop for multiple times to get convergence
# The trained model then obtained is used to label the test files.
from json import load, dump
from os import listdir
from sys import exit, argv
from math import log10
from re import sub
from random import randint, choice

# Stopwords Reference: https://github.com/Alir3z4/stop-words/blob/0e438af98a88812ccc245cf31f93644709e70370/english.txt
# Words to be ignored while learning
STOPWORDS = {'all', 'whys', 'being', 'over', 'isnt', 'through', 'yourselves', 'hell', 'its', 'before', 'wed', 'with',
             'had', 'should', 'to', 'lets', 'under', 'ours', 'has', 'ought', 'do', 'them', 'his', 'very', 'cannot',
             'they', 'werent', 'not', 'during', 'yourself', 'him', 'nor', 'wont', 'did', 'theyre', 'this', 'she', 'up',
             'each', 'havent', 'where', 'shed', 'because', 'doing', 'theirs', 'some', 'whens', 'are', 'further', 'we',
             'ourselves', 'out', 'what', 'for', 'heres', 'while', 'does', 'above', 'between', 'youll', 'be', 'who',
             'were', 'here', 'hers', 'by', 'both', 'about', 'would', 'wouldnt', 'didnt', 'ill', 'against', 'arent',
             'youve', 'theres', 'or', 'thats', 'weve', 'own', 'whats', 'dont', 'into', 'youd', 'whom', 'down', 'doesnt',
             'theyd', 'couldnt', 'your', 'from', 'her', 'hes', 'there', 'only', 'been', 'whos', 'hed', 'few', 'too',
             'themselves', 'was', 'until', 'more', 'himself', 'on', 'but', 'you', 'hadnt', 'shant', 'mustnt', 'herself',
             'than', 'those', 'he', 'me', 'myself', 'theyve', 'these', 'cant', 'below', 'of', 'my', 'could', 'shes',
             'and', 'ive', 'then', 'wasnt', 'is', 'am', 'it', 'an', 'as', 'itself', 'im', 'at', 'have', 'in', 'any',
             'if', 'again', 'hasnt', 'theyll', 'no', 'that', 'when', 'same', 'id', 'how', 'other', 'which', 'shell',
             'shouldnt', 'our', 'after', 'most', 'such', 'why', 'wheres', 'a', 'hows', 'off', 'i', 'youre', 'well',
             'yours', 'their', 'so', 'the', 'having', 'once'}


def calc_p_T(dc):  # Calculate p(Topic) or probability of a topic
    dc.p_T = {}
    for topic in dc.docCountInTopic:
        dc.p_T[topic] = dc.docCountInTopic[topic] * 1.0 / dc.totalDocCount
    return dc


def calc_p_word_in_topic(dc):  # Calculate p(w/Topic) for all words
    dc.p_word_in_topic = {}
    for word in dc.p_w_t.keys():
        dc.p_word_in_topic[word] = {}
        for topic in dc.p_w_t[word].keys():
            dc.p_word_in_topic[word][topic] = (dc.p_w_t[word][topic] * 1.0) / dc.words_in_topic[topic]
    return dc


def write_to_file(file_name, writeData):  # Write the learned model
    with open(file_name, "w+") as f:
        dump(writeData, f, indent=2)


def print_matrix(topics, confusion_matrix):  # Print the confusion matrix
    print '            ',
    for t in topics:
        print '%-12s' % t,
    print ''
    for actual_topic in topics:
        print '%-12s' % actual_topic,
        for test_topic in topics:
            if actual_topic in confusion_matrix:
                if test_topic in confusion_matrix[actual_topic]:
                    print '%-12i' % confusion_matrix[actual_topic][test_topic],
                else:
                    print '%-12i' % 0,
            else:
                for i in range(20):
                    print '%-12i' % 0,
        print ''


def testData(words, probabilityTable, p_T):
    # Function tests a document uisng the learned model and returns the proposed label
    # We a used a minimum value of 10 raise to -6 for the missing words in the learned model.
    # For fraction 0.0 there will be no maximum and hence the else part will be used and we randomly select the topic.
    p = {}
    for topic in p_T:
        p[topic] = 0
        for word in words:
            if word in probabilityTable:
                if topic in probabilityTable[word]:
                    p[topic] += log10(probabilityTable[word][topic])
                else:
                    p[topic] += -6
            else:
                p[topic] += -6
    if len(p) > 0:
        return max(p.iterkeys(), key=(lambda k: p[k]))
    else:
        return choice(get_dir_contents("train"))


def get_dir_contents(directory):
    # Get list of files or directory in input directories
    # Assumption the files starting with '.' are not to be used for learning
    topics = listdir(directory)  # list of all topics
    for word in topics[:]:
        if word.startswith('.'):
            topics.remove(word)
    return topics


def get_words(content):  # get list of all words in file except STOPWORDS
    return [word for word in sub('[^a-zA-Z \n]', '', content).lower().split() if word not in STOPWORDS]


def validate_initialize():  # Validate command line input and initialize variables
    fraction = 0  # initialize fraction to 0
    if len(argv) > 5:  # check if too many arguments
        print("Too many arguments!")
        exit(1)
    if len(argv) < 4:  # check if too few arguments
        print("Too few arguments!")
        exit(1)
    mode = argv[1]
    dataset_directory = argv[2]
    model_file = argv[3]
    if mode == "train":
        if len(argv) == 5:
            try:
                fraction = float(argv[4])
                if fraction < 0 or fraction > 1:  # check invalid fraction
                    print("Fraction should be a number between 0.0 and 1.0")
                    exit(1)
            except:  # invalid fraction
                print("Fraction should be a number between 0.0 and 1.0")
                exit(1)
        else:  # invalid input
            print("Train mode takes 4 command line arguments")
            exit(1)
    elif mode == "test":
        if len(argv) != 4:
            print("Test mode takes 3 command line arguments")
            exit(1)
    else:  # invalid mode
        print("Invalid mode!")
        exit(1)
    bias = fraction * 100
    topics = get_dir_contents(dataset_directory)
    return bias, mode, dataset_directory, model_file, topics

class DocumentClassification:
	# Initialize all the dictionaries to be used in training model.
    def __init__(self):
        self.docCountInTopic = {}
        self.words_in_topic = {}
        self.p_w_t = {}
        self.totalDocCount = 0
        pass

    # After each word is labelled this function is called to update all the count dictionaries.
    def processDocument(self, words, topic):
        if topic in self.docCountInTopic:
            self.docCountInTopic[topic] += 1
        else:
            self.docCountInTopic[topic] = 1
        if topic in self.words_in_topic:
            self.words_in_topic[topic] += len(words)
        else:
            self.words_in_topic[topic] = len(words)
        for word in words:
            if word in self.p_w_t:
                if topic in self.p_w_t[word]:
                    self.p_w_t[word][topic] += 1
                else:
                    self.p_w_t[word][topic] = 1
            else:
                self.p_w_t[word] = {topic: 1}
        self.totalDocCount += 1


if __name__ == "__main__":
    bias, mode, dataset_directory, model_file, topics = validate_initialize()
    if mode == 'train':
        print "Training...\nIt may take a few minutes."
        dc = DocumentClassification()
        unknownList = dict()
        for topic in topics:
            documents = get_dir_contents(dataset_directory + "/" + topic)
            for document in documents:
                filename_with_path = (dataset_directory + "/" + topic + "/" + document)
                with open(filename_with_path) as f:
                    content = f.read()
                # flip a coin according to fraction
                flip = randint(0, 100)
                if flip > bias or (flip == 0 and bias == 0):  # topic  = unknown if flip > bias
                    unknownList[dataset_directory + "/" + topic + "/" + document] = get_words(content)
                    continue
                words = get_words(content)
                dc.processDocument(words, topic)
        dc = calc_p_word_in_topic(dc)
        dc = calc_p_T(dc)
        if len(unknownList) != 0:  # semi-supervised / unsupervised
            if bias == 0:  # unsupervised
                for i in range(10):  # Learn model iteratively such that it may start converging after a few iterations
                    dc.__init__()  # reinitialize just the frequency variables not the learned model
                    for unknown_file in unknownList:
                        label = testData(unknownList[unknown_file], dc.p_word_in_topic, dc.p_T)
                        dc.processDocument(unknownList[unknown_file], label)
                    # Overwrite the model with the new model in the iteration
                    dc = calc_p_word_in_topic(dc)
                    dc = calc_p_T(dc)
            else:  # For semi-supervised learning
                # Label the unlabeled documents using the model learned until now
                for unknown_file in unknownList:
                    label = testData(unknownList[unknown_file], dc.p_word_in_topic, dc.p_T)
                    dc.processDocument(unknownList[unknown_file], label)
                # Overwrite the model with the new model in the iteration
                dc = calc_p_word_in_topic(dc)
                dc = calc_p_T(dc)
        # Write model to file
        writeData = {
            'p_word_in_topic': dc.p_word_in_topic,
            'p_T': dc.p_T,
        }
        write_to_file(model_file, writeData)
        top_words_per_topic = {}
        for topic in topics:
            top_words_per_topic[topic] = sorted(
                [[dc.p_w_t[word][t], word] for word in dc.p_w_t for t in dc.p_w_t[word] if t == topic],
                key=lambda element: element[0], reverse=True)[:10]
            top_words_per_topic[topic] = [ element[1] for element in top_words_per_topic[topic]]
        write_to_file("distinctive_words.txt", top_words_per_topic)
    else:  # test mode
        print "Testing...\nIt may take a few minutes."
        # read the learned model
        count_doc = 0  # Count of Documents tested
        accuracy = 0  # Count of Documents labelled correctly
        try:
            with open(model_file) as f:
                readData = load(f)  # read model
            p_T = readData['p_T']
            p_word_in_topic = readData['p_word_in_topic']
        except:
            print "Invalid Model File!"
            exit(1)
        confusion_matrix = {}
        for topic in topics:
            confusion_matrix[topic] = {}
            documents = get_dir_contents(dataset_directory + "/" + topic)
            for document in documents:
                filename_with_path = (dataset_directory + "/" + topic + "/" + document)
                with open(filename_with_path) as f:
                    content = f.read()
                label = str(testData(get_words(content), p_word_in_topic, p_T))
                if label in confusion_matrix[topic]:
                    confusion_matrix[topic][label] += 1
                else:
                    confusion_matrix[topic][label] = 1
                count_doc += 1
                if topic == label:
                    accuracy += 1
        print "Accuracy: ", (accuracy * 1.0 / count_doc) * 100
        print_matrix(topics, confusion_matrix)
