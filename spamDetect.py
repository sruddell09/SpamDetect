import sys
from math import log

def process_line(line):
    label, subject_line = line.strip().split('\t') 
    words = subject_line.split()

    return label, words

def update_counts(label, words, dict_class, dict_spam, dict_ham):
    # 1. update dict_class
    if not label in dict_class:
        dict_class[label] = 1.0
    else:
        dict_class[label] += 1
    # 2. update dict_spam
    if label == '1':
        for word in words:
            if not word in dict_spam:
                dict_spam[word] = 1.0
            else:
                dict_spam[word] += 1                    
    # 3. update dict_ham
    elif label == '0':
        for word in words:
            if not word in dict_ham:
                dict_ham[word] = 1.0
            else:
                dict_ham[word] += 1  

    return dict_class, dict_spam, dict_ham

def smooth(dict_spam, dict_ham):
    #Define unknowns
    dict_spam['unknown'] = 0.0
    dict_ham['unknown'] = 0.0
    #Add-one smoothing
    for word in dict_spam:
        dict_spam[word] += 1
    for word in dict_ham:
        dict_ham[word] += 1
    
    return dict_spam, dict_ham

def normalize(dict_spam, dict_ham, ds_tokens, dh_tokens):
    #Convert counts into probabilities (posterior = liklihood * prior)
    for word in dict_spam:
        dict_spam[word] = dict_spam[word] / (ds_tokens + len(dict_spam))
    for word in dict_ham:
        dict_ham[word] = dict_ham[word] / (dh_tokens + len(dict_ham))
        
    return dict_spam, dict_ham

def argmax(words, dict_class, dict_spam, dict_ham):
    #Initializing the scores
    score_spam = log(1)
    score_ham = log(1)

    #Total number of documents in the collection
    doc_coll = dict_class['0'] + dict_class['1']
    
    #P(c) = number of documents for a given class / total number of documents
    score_spam += log(dict_class['1'] / doc_coll)
    score_ham += log(dict_class['0'] / doc_coll)

    #P(w|c) =
    for word in words:
        if word not in dict_spam:
            score_spam += log(dict_spam['unknown'])
        else:
            score_spam += log(dict_spam[word])
        if word not in dict_ham:
            score_ham += log(dict_ham['unknown'])
        else:
            score_ham += log(dict_ham[word])

    #Label as spam or ham
    if score_spam > score_ham:
        return '1'
    else:
        return '0'

if __name__ == '__main__':

    ###TRAINING###
    
    # define empty dictionaries: 1 for class counts and 1 for each class containing a bag of words
    dict_class = {}
    dict_spam = {}
    dict_ham = {}

    lines = sys.stdin.readlines()
    for line in lines:
        #strip and split to extract the bag of words along with its label
        label, words = process_line(line)
        #update counts of all dictionaries 
        dict_class, dict_spam, dict_ham = update_counts(label, words, dict_class, dict_spam, dict_ham)
    
    #store token values before smoothing
    ds_tokens = sum(dict_spam.values())
    dh_tokens = sum(dict_ham.values())

    #smooth and normalize
    dict_spam, dict_ham = smooth(dict_spam, dict_ham)
    dict_spam, dict_ham = normalize(dict_spam, dict_ham, ds_tokens, dh_tokens)
    
    ###TESTING###

    TP = 0.0 #True positives
    FP = 0.0 #False positives
    TN = 0.0 #True negatives
    FN = 0.0 #False negatives

    f = open("spam_assassin.test")
    for line in f:
        t_label, words = process_line(line)
        c_label = argmax(words, dict_class, dict_spam, dict_ham)
        if c_label == '1' and t_label == '1':
            TP += 1
        if c_label == '1' and t_label != '1':
            FP += 1
        if c_label != '1' and t_label != '1':
            TN += 1
        if c_label != '1' and t_label == '1':
            FN += 1
    f.close()

    print "Precision: %f" % (TP / (TP + FP))
    print "Recall: %f" % (TP / (TP + FN))



            


    
    

    

