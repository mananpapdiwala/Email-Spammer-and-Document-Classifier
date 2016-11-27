# Usage:
#   ./topics mode dataset-directory model-file [fraction]
import json
import os
import sys
import math
from random import randint

import re

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

def testData(words, probabilityTable, p_T, defaultDict):
    probability = 1
    p = {}
    #words = set(words).intersection(probabilityTable.keys())
    words = set(words)
    words = [word for word in words if word not in STOPWORDS]
    for topic in p_T:
        #p[topic] = math.log10(p_T[topic])
        p[topic] = 0
        for word in words:
			if word in probabilityTable:	
				if topic in probabilityTable[word]:
					p[topic] += math.log10(probabilityTable[word][topic])
				else:
					#p[topic] += math.log10(1.0/defaultDict[topic])
					p[topic] += -6
			else:
				#p[topic] += math.log10(1.0/defaultDict[topic])
				p[topic] += -6
    return max(p.iterkeys(), key=(lambda k: p[k]))

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
        bias = fraction * 100
        docCountInTopic = {}
        totalDocCount = 0
        p_w_t = {}
        words_in_topic = {}
        count_wordDoc_per_topic = {}
        topics = os.listdir(dataset_directory)  # list of all topics
        for word in topics[:]:
            if word.startswith('.'):
                topics.remove(word)
        for topic in topics:
            docCountInTopic[topic] = 0
        for topic in topics:
            documents = os.listdir(dataset_directory + "/" + topic)
            for word in documents[:]:
                if word.startswith('.'):
                    documents.remove(word)
            for document in documents:
                filename_with_path = (dataset_directory + "/" + topic + "/" + document)
                with open(filename_with_path) as f:
                    content = f.read()
                # flip a coin according to fraction
                flip = randint(0, 100)
                if flip > bias:  # topic  = unknown if flip > bias
                    filename_with_path = (dataset_directory + "/" + "unknown" + "/" + document)
                    if 'unknown' in docCountInTopic:
                        docCountInTopic['unknown'] += 1
                    else:
                        docCountInTopic['unknown'] = 1
                else:
                    docCountInTopic[topic] += 1

                words = re.sub('[^a-zA-Z \n]', '', content).lower().split()
                #words = [word for word in words if word not in STOPWORDS]
                if topic in words_in_topic:
                    words_in_topic[topic] += len(words)
                else:
                    words_in_topic[topic] = len(words)
                for word in words:
                    if word in p_w_t:
                        if topic in p_w_t[word]:
                            p_w_t[word][topic] += 1
                        else:
                            p_w_t[word][topic] = 1
                    else:
                        p_w_t[word] = {topic: 1}
                wordSet = set(words)
                for word in wordSet:
                    if word in count_wordDoc_per_topic:
                        if topic in count_wordDoc_per_topic[word]:
                            count_wordDoc_per_topic[word][topic] += 1
                        else:
                            count_wordDoc_per_topic[word][topic] = 1
                    else:
                        count_wordDoc_per_topic[word] = {topic: 1}
                totalDocCount += 1

        p_word_in_topic = {}
        for word in p_w_t.keys():
            p_word_in_topic[word] = {}
            for topic in p_w_t[word].keys():
                p_word_in_topic[word][topic] = (p_w_t[word][topic] * 1.0) / words_in_topic[topic]

        for word in p_w_t:
            for topic in p_w_t[word].keys():
                p_w_t[word][topic] = p_w_t[word][topic] * 1.0 / (
                    docCountInTopic[topic] if docCountInTopic[topic] != 0 else 1)

        p_count_wordDoc_per_topic = {}
        for word in count_wordDoc_per_topic.keys():
            p_count_wordDoc_per_topic[word] = {}
            for topic in count_wordDoc_per_topic[word].keys():
                p_count_wordDoc_per_topic[word][topic] = (count_wordDoc_per_topic[word][topic] * 1.0) / (
                    docCountInTopic[topic] if docCountInTopic[topic] != 0 else 1)

        p_T = {}
        for topic in docCountInTopic:
            p_T[topic] = docCountInTopic[topic] * 1.0 / totalDocCount

        # Write model to file
        writeData = {
            'p_w_t': p_w_t,
            'p_word_in_topic': p_word_in_topic,
            'p_count_wordDoc_per_topic': p_count_wordDoc_per_topic,
            'p_T': p_T,
            'docCountInTopic' : docCountInTopic,
            'words_in_topic' : words_in_topic
            }
        with open(model_file, "w+") as f:
            json.dump(writeData, f, indent=2)
        f.close()
    else:  # test mode
        # read the learned model
        count_doc = 0
        count_accuracy_p2 = 0
        count_accuracy_p3 = 0
        with open(model_file) as f:
            readData = json.load(f)
        p_T = readData['p_T']
        p_word_in_topic = readData['p_word_in_topic']
        p_count_wordDoc_per_topic = readData['p_count_wordDoc_per_topic']
        p_w_t = readData['p_w_t']
        docCountInTopic = readData['docCountInTopic']
        words_in_topic = readData['words_in_topic']

        topics = os.listdir(dataset_directory)  # list of all topics
        for word in topics[:]:
            if word.startswith('.'):
                topics.remove(word)
        for topic in topics:
            documents = os.listdir(dataset_directory + "/" + topic)
            for word in documents[:]:
                if word.startswith('.'):
                    documents.remove(word)
            for document in documents:
                filename_with_path = (dataset_directory + "/" + topic + "/" + document)
                with open(filename_with_path) as f:
                    content = f.read()
                words = re.sub('[^a-zA-Z \n]', '', content).lower().split()
                #p1 = testData(words, p_w_t, p_T, {})
                p2 = testData(words, p_count_wordDoc_per_topic, p_T, docCountInTopic)
                p3 = testData(words, p_word_in_topic, p_T, words_in_topic)
                count_doc += 1
                if topic == p2: count_accuracy_p2 += 1
                if topic == p3: count_accuracy_p3 += 1
        f.close()
        print "Accuracy for p2: ", (count_accuracy_p2*1.0/count_doc)*100
        print "Accuracy for p3: ", (count_accuracy_p3*1.0/count_doc)*100