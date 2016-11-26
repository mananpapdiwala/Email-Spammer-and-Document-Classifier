# Usage:
#   ./topics mode dataset-directory model-file [fraction]

import json
import os
import sys
from random import randint
import re
import math

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
        docCountInTopic = {}
        topics = os.listdir(dataset_directory)  # list of all topics
        for word in topics[:]:
            if word.startswith('.'):
                topics.remove(word)
        for topic in topics:
            docCountInTopic[topic] = 0
        p_w_t = {}
        words_in_topic = {}
        count_wordDoc_per_topic = {}
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
                else:
                    docCountInTopic[topic] += 1

                words = re.sub('[^a-zA-Z \n]', '', content).lower().split()

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
        for word in p_w_t:
            for topic in p_w_t[word].keys():
                p_w_t[word][topic] = p_w_t[word][topic] * 1.0 / (docCountInTopic[topic] if docCountInTopic[topic] != 0 else 1)

        p_word_in_topic = {}
        for word in p_w_t.keys():
        	p_word_in_topic[word] = {}
        	for topic in p_w_t[word].keys():
        		p_word_in_topic[word][topic] = (p_w_t[word][topic] * 1.0)/words_in_topic[topic]

        p_count_wordDoc_per_topic = {}
        for word in count_wordDoc_per_topic.keys():
        	p_count_wordDoc_per_topic[word] = {}
        	for topic in count_wordDoc_per_topic[word].keys():
        		p_count_wordDoc_per_topic[word][topic] = (count_wordDoc_per_topic[word][topic] * 1.0) / (docCountInTopic[topic] if docCountInTopic[topic] != 0 else 1)

        with open(model_file, "w+") as f:
        	json.dump(p_w_t, f)
        	f.write("\n")
        	json.dump(p_word_in_topic, f)
        	f.write("\n")
        	json.dump(p_count_wordDoc_per_topic, f)
        f.close()
        x = 0
    else:  # test mode
        pass
