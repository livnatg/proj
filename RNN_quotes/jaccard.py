# Imports
import nltk.corpus
import nltk.tokenize.punkt
import nltk.stem.snowball
import string

# Get default English stopwords and extend with punctuation
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')

# Create tokenizer and stemmer
# tokenizer = nltk.sent_tokenize()

def is_ci_token_stopword_set_match(a, b, threshold=0.5):
    """Check if a and b are matches."""
    print a
    print nltk.sent_tokenize(a)
    tokens_a = [token.lower().strip(string.punctuation) for token in nltk.word_tokenize(a) \
                    if token.lower().strip(string.punctuation) not in stopwords]
    print tokens_a
    tokens_b = [token.lower().strip(string.punctuation) for token in nltk.word_tokenize(b) \
                    if token.lower().strip(string.punctuation) not in stopwords]

    # Calculate Jaccard similarity
    ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
    print ratio
    return (ratio >= threshold)

f = open('data/data.csv', 'rb')
for line in f:
    is_ci_token_stopword_set_match(line,"changed ordinary understand opposition necessary formed inevitable lose necessary whole delay do written everything fortune hides coward language find income sum")