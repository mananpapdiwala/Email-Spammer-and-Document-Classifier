# Usage:
#   ./topics mode dataset-directory model-file [fraction]

import json
import os
import sys
from random import randint
import re
import math

# Words to be ignored while learning
STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
             'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
             'needn', 'shan', 'shouldn', 'wasn', 'weren', 'wouldn', 'arent', 'ive', 'dont', 'first', 'could'}


# Convert set obj to list for writing json into file
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


# Function returns dictionary with words as keys and their frequency in a doc as the values
def frequency(words):
    wordCount = {}
    for word in words:
        if word in wordCount:
            wordCount[word] += 1
        else:
            wordCount[word] = 1
    return wordCount


# Total no. of words in a document
def wordCount(words):
    return sum(words.values())


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
            print("Invalid Input!")
            sys.exit(1)
    elif mode == "test":
        if len(sys.argv) != 4:
            print("Invalid Input!")
            sys.exit(1)
    else:  # invalid mode
        print("Invalid mode!")
        sys.exit(1)

    if mode == 'train':
        bias = fraction * 100
        topics = os.listdir(dataset_directory)  # list of all topics
        for word in topics[:]:
            if word.startswith('.'):
                topics.remove(word)

        wordList = {}
        tf = {}
        numarray = {}
        puncNumCount = 0
        for topic in topics:
            documents = os.listdir(dataset_directory + "/" + topic)
            print topic
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

                words = re.sub('[^a-zA-Z \n]', '', content).lower().split()
                words = set(words) - STOPWORDS
                # remove single letter words
                for word in words.copy():
                    if len(word) < 2:
                        words.remove(word)

                for num in content.split():
                    if str(re.search('[a-zA-Z]', num)) == 'None':
                        numarray[topic] = 1 if topic not in numarray else numarray[topic] + 1
                        #puncNumCount += 1

                tf[filename_with_path] = frequency(words)
                for word in tf[filename_with_path]:
                    tf[filename_with_path][word] *= 1.0 / wordCount(tf[filename_with_path])
                    wordList[word] = 1 if word not in wordList else wordList[word] + 1
        # print wordList
        no_of_docs = len(tf)
        for doc in tf:
            for word in tf[doc]:
                x = wordList[word]
                tf[doc][word] *= math.log(no_of_docs * 1.0 / (1 + x))  # tf*idf

        impWords = {}
        for doc in tf:
            max1 = max([[tf[doc][key], key] for key in tf[doc].keys()])
            label = doc.split('/')[1]
            if label not in impWords:
                impWords[label] = {max1[1]: max1[0]}
            else:
                impWords[label].update({max1[1]: max1[0]})

            tf[doc].pop(max1[1])
            max1 = max([[tf[doc][key], key] for key in tf[doc].keys()])
            label = doc.split('/')[1]
            if label not in impWords:
                impWords[label] = {max1[1]: max1[0]}
            else:
                impWords[label].update({max1[1]: max1[0]})

            tf[doc].pop(max1[1])
            max1 = max([[tf[doc][key], key] for key in tf[doc].keys()])
            label = doc.split('/')[1]
            if label not in impWords:
                impWords[label] = {max1[1]: max1[0]}
            else:
                impWords[label].update({max1[1]: max1[0]})

        print impWords

        with open("model.txt", "w+") as f:
            json.dump(impWords, f, default=set_default)
        f.close()
        with open("model.txt", 'rb') as f:
            my_list = json.load(f)
        f.close()
        print my_list
        with open("test/hockey/53925") as f:
            newWords = re.sub('[^a-zA-Z \n]', '', f.read()).lower().split()
        for topic in my_list:
            for word in my_list[topic]:
                if word in newWords:
                    print topic, word
    else:  # test mode
        pass
