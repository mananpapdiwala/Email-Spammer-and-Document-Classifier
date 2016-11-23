import sys
import time
import os
import re


def getFileList(x_dataset):
    spam_file_list = os.listdir(x_dataset + "/spam")
    non_spam_file_list = os.listdir(x_dataset + "/notspam")
    return {"spam": spam_file_list, "notspam": non_spam_file_list}


def getFileData(x_all_emails_list, x_dataset):
    # i = 0
    spam_files = [{}, 0]
    not_spam_files = [{}, 0]
    for file_name in x_all_emails_list["spam"]:
        spam_files = readFile(x_dataset + "/spam/" + file_name, spam_files)
        # i += 1

    # i = 0
    for file_name in x_all_emails_list["notspam"]:
        not_spam_files = readFile(x_dataset + "/notspam/" + file_name, not_spam_files)
        # i += 1
    return {"spam": spam_files, "notspam": not_spam_files}


def readFile(file_name, current_data):
    words = current_data[0]
    total_no_of_words = 0
    file_data = open(file_name, 'r')

    for line in file_data:
        for word in line.split():
            if str(re.search('[a-zA-Z]', word)) == "None":
                continue
            word = re.sub('[^a-zA-Z0-9@]', '', word)
            total_no_of_words += 1
            if words.has_key(word):
                words[word] += 1
            else:
                words[word] = 1
    return [words, current_data[1] + total_no_of_words]


def getProbabilityDistribution(words, email_list):
    spam_count = words["spam"][1]
    not_spam_count = words["notspam"][1]
    total_words = spam_count + not_spam_count
    probability_of_spam_words = {}
    for word in words["spam"][0]:
        probability_of_spam_words[word] = (1.0*words["spam"][0][word])/(spam_count)

    probability_of_notspam_words = {}
    for word in words["notspam"][0]:
    	probability_of_notspam_words[word] = (1.0*words["notspam"][0][word]/not_spam_count)
    

    return [probability_of_spam_words, probability_of_notspam_words]

def writeProbabilityInFile(spam, notspam, filePath):
	file = open(filePath, 'w')
	for word in spam:
		file.write('%s,%s'%(word, spam[word]))
		file.write(' ')
	file.write('\n')
	for word in notspam:
		file.write('%s,%s'%(word, notspam[word]))
		file.write(' ')
	file.close()

start_time = time.time()
print time.asctime(time.localtime(time.time()))

(mode, technique, dataset, modelfile) = sys.argv[1:]

if mode == "train":
	email_directories = getFileList(dataset)
	word_set = getFileData(email_directories, dataset)
	[spam_probability, notspam_probability] = getProbabilityDistribution(word_set, email_directories)
	writeProbabilityInFile(spam_probability, notspam_probability, modelfile);

end_time = time.time()
print time.asctime(time.localtime(time.time()))
print end_time - start_time