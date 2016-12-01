# Usage:
#   ./topics mode dataset-directory model-file [fraction]
import json
import os
import sys
import math

import re

import time
from random import randint, choice

# Words to be ignored while learning
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


def testData(words, probabilityTable, p_T):
    p = {}
    for topic in p_T:
        p[topic] = 0
        words = [word for word in words if word not in STOPWORDS]
        for word in words:
            if word in probabilityTable:
                if topic in probabilityTable[word]:
                    p[topic] += math.log10(probabilityTable[word][topic])
                else:
                    p[topic] += -6
            else:
                p[topic] += -6
    if len(p) > 0:
        return max(p.iterkeys(), key=(lambda k: p[k]))
    else:
        return choice(get_dir_contents("train"))


def get_dir_contents(directory):
    topics = os.listdir(directory)  # list of all topics
    for word in topics[:]:
        if word.startswith('.'):
            topics.remove(word)
    return topics


def get_words(content):
    return re.sub('[^a-zA-Z \n]', '', content).lower().split()


class DocumentClassification:
    def __init__(self):
        self.docCountInTopic = {}
        self.words_in_topic = {}
        self.p_w_t = {}
        self.count_wordDoc_per_topic = {}
        self.totalDocCount = 0
        pass

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
    fraction = 0  # initialize fraction to 0
    if len(sys.argv) > 5:  # check if too many arguments
        print("Too many arguments!")
        sys.exit(1)
    if len(sys.argv) < 4:  # check if too few arguments
        print("Too few arguments!")
        sys.exit(1)
    mode = sys.argv[1]
    dataset_directory = sys.argv[2]
    model_file = sys.argv[3]
    if mode == "train":
        if len(sys.argv) == 5:
            try:
                fraction = float(sys.argv[4])
                if fraction < 0 or fraction > 1:  # check invalid fraction
                    print("Fraction should be a number between 0.0 and 1.0")
                    sys.exit(1)
            except:  # invalid fraction
                print("Fraction should be a number between 0.0 and 1.0")
                sys.exit(1)
        else:  # invalid input
            print("Train mode takes 4 command line arguments")
            sys.exit(1)
    elif mode == "test":
        if len(sys.argv) != 4:
            print("Test mode takes 3 command line arguments")
            sys.exit(1)
    else:  # invalid mode
        print("Invalid mode!")
        sys.exit(1)

    if mode == 'train':
        start_time = time.time()
        dc = DocumentClassification()
        bias = fraction * 100
        topics = get_dir_contents(dataset_directory)
        unknownList = dict()
        for topic in topics:
            documents = get_dir_contents(dataset_directory + "/" + topic)
            for document in documents:
                filename_with_path = (dataset_directory + "/" + topic + "/" + document)
                with open(filename_with_path) as f:
                    content = f.read()
                # flip a coin according to fraction
                flip = randint(0, 100)
                if flip > bias:  # topic  = unknown if flip > bias
                    unknownList[dataset_directory + "/" + topic + "/" + document] = get_words(content)
                    continue
                words = [word for word in get_words(content) if word not in STOPWORDS]
                dc.processDocument(words, topic)

        dc.p_word_in_topic = {}
        for word in dc.p_w_t.keys():
            dc.p_word_in_topic[word] = {}
            for topic in dc.p_w_t[word].keys():
                dc.p_word_in_topic[word][topic] = (dc.p_w_t[word][topic] * 1.0) / dc.words_in_topic[topic]

        dc.p_T = {}
        for topic in dc.docCountInTopic:
            dc.p_T[topic] = dc.docCountInTopic[topic] * 1.0 / dc.totalDocCount

        if len(unknownList) != 0:
            for unknown_file in unknownList:
                label = testData(unknownList[unknown_file], dc.p_word_in_topic, dc.p_T)
                dc.processDocument(unknownList[unknown_file], label)

            dc.p_word_in_topic = {}
            for word in dc.p_w_t.keys():
                dc.p_word_in_topic[word] = {}
                for topic in dc.p_w_t[word].keys():
                    dc.p_word_in_topic[word][topic] = (dc.p_w_t[word][topic] * 1.0) / dc.words_in_topic[topic]

            dc.p_T = {}
            for topic in dc.docCountInTopic:
                dc.p_T[topic] = dc.docCountInTopic[topic] * 1.0 / dc.totalDocCount
        # Write model to file
        writeData = {
            'p_word_in_topic': dc.p_word_in_topic,
            'p_T': dc.p_T,
        }
        with open(model_file, "w+") as f:
            json.dump(writeData, f, indent=2)
        top_words_per_topic = {}
        for topic in topics:
            top_words_per_topic[topic] = sorted(
                [[dc.p_w_t[word][t], word] for word in dc.p_w_t for t in dc.p_w_t[word] if
                 word not in STOPWORDS and t == topic], key=lambda element: element[0], reverse=True)[:10]
        with open("distinctive_words.txt", "w+") as f:
            json.dump(top_words_per_topic, f, indent=4)
        print time.time() - start_time
    else:  # test mode
        # read the learned model
        count_doc = 0
        count_accuracy_p3 = 0
        try:
            with open(model_file) as f:
                readData = json.load(f)
            p_T = readData['p_T']
            p_word_in_topic = readData['p_word_in_topic']
        except:
            print "Invalid Model File!"
            sys.exit(1)
        topics = get_dir_contents(dataset_directory)
        for topic in topics:
            documents = get_dir_contents(dataset_directory + "/" + topic)
            for document in documents:
                filename_with_path = (dataset_directory + "/" + topic + "/" + document)
                with open(filename_with_path) as f:
                    content = f.read()
                p3 = testData(get_words(content), p_word_in_topic, p_T)
                count_doc += 1
                if topic == p3:
                    count_accuracy_p3 += 1
        print "Accuracy for p3: ", (count_accuracy_p3 * 1.0 / count_doc) * 100
