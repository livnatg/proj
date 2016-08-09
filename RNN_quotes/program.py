import csv
import itertools
import numpy as np
import nltk
from rnn_model import Simpel_rnn
from rnn_lstm import LSTM_rnn
from datasketch import MinHash
import string

vocabulary_size = 1500
model_type = "LSTM"
# nltk.download()
print "Reading CSV file..."
with open('data/data.csv', 'rb') as f: #reddit-comments-2015-08
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # quotes to sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower())for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % ("SENTENCE_START", x, "SENTENCE_END") for x in sentences]
print (str(len(sentences))+" sentences" )
# Tokenize the sentences into words
tokens = [nltk.word_tokenize(sen) for sen in sentences]
# print tokens
# Count the word frequencies
frequency = nltk.FreqDist(itertools.chain(*tokens))
print (str(len(frequency.items()))+ " words")
# Build the data matrix
vocabularyList = frequency.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocabularyList]
index_to_word.append("UNKNOWN_TOKEN")
word_to_index = dict([(j,i) for i,j in enumerate(index_to_word)])
# Replace the rest of the words to unknown token
for i, sen in enumerate(tokens):
    tokens[i] = [j if j in word_to_index else "UNKNOWN_TOKEN" for j in sen]
X_train = np.asarray([[word_to_index[j] for j in tok[:-1]] for tok in tokens])
Y_train = np.asarray([[word_to_index[j] for j in tok[1:]] for tok in tokens])

if (model_type=="LSTM"):
    model = LSTM_rnn(vocabulary_size)
    print "LSTM"
else:
    model = Simpel_rnn(vocabulary_size)
    print "SIMPLE"

print "Random Loss:" + str(np.log(vocabulary_size))
np.random.seed(10)
# //3293
losses = model.totalSGD( X_train[:100], Y_train[:100])
num_sentences = 10
# num_words = 3
newSentence = []
for i in range(num_sentences):
    newSentence.append(model.getSentence(word_to_index,index_to_word))

# print(len(newSentence))
# print (newSentence)
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')
for sen in newSentence:
    data1 = [token.lower().strip(string.punctuation) for token in nltk.word_tokenize(sen) \
                    if token.lower().strip(string.punctuation) not in stopwords]
    f = open('data/data.csv', 'rb')
    for line in f:
        data2 = [token.lower().strip(string.punctuation) for token in nltk.word_tokenize(line) \
                        if token.lower().strip(string.punctuation) not in stopwords]
        m1, m2 = MinHash(), MinHash()
        for d in data1:
            m1.update(d.encode('utf8'))
        for d in data2:
            m2.update(d.encode('utf8'))
        # print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))

        s1 = set(data1)
        s2 = set(data2)
        actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))

        if(actual_jaccard > 0.3):
            print("Actual Jaccard for data1 and data2 is", actual_jaccard)
            print sen
            print line