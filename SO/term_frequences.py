import operator 
import json
from collections import Counter
import preprocess
from nltk.corpus import stopwords
import string
from nltk import bigrams 
from collections import defaultdict
import math
 
com = defaultdict(lambda : defaultdict(int))
fname = 'tweetset.json'
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']
print(stop) 

with open(fname, 'r') as f:
    count_all = Counter()
    for line in f:
        tweet = json.loads(line)
       # Count terms only (no mentions)
        terms_only = [term for term in preprocess.preprocess(tweet['text'], lowercase=True) 
                    if term not in stop and
                    not term.startswith(('@', 'RT'))]                    
        count_all.update(terms_only)

        # Build co-occurrence matrix
        for i in range(len(terms_only)-1):            
            for j in range(i+1, len(terms_only)):
                w1, w2 = sorted([terms_only[i], terms_only[j]])                
                if w1 != w2:
                    com[w1][w2] += 1
    
    # Print the first 5 most frequent words
    print(count_all.most_common(5))


# n_docs is the total number of tweets
p_t = {}
p_t_com = defaultdict(lambda : defaultdict(int))
n_docs = 1400

#computing the probability of a word
for term, n in count_all.items():
    p_t[term] = n / n_docs
    for t2 in com[term]:
        p_t_com[term][t2] = com[term][t2] / n_docs


positive_vocab = [
    'good', 'nice', 'great', 'awesome', 'fantastic', 'terrific',
    'victory', 'like', 'love'
]
negative_vocab = [
    'bad', 'terrible', 'crap', 'useless', 'hate', 'dislike'
]

pmi = defaultdict(lambda : defaultdict(int))
for t1 in p_t:
    for t2 in com[t1]:
        denom = p_t[t1] * p_t[t2]
        pmi[t1][t2] = math.log2(p_t_com[t1][t2] / denom)
 
semantic_orientation = {}
for term, n in p_t.items():
    positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
    negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
    semantic_orientation[term] = positive_assoc - negative_assoc

semantic_sorted = sorted(semantic_orientation.items(), 
                         key=operator.itemgetter(1), 
                         reverse=True)
top_pos = semantic_sorted[:50]
top_neg = semantic_sorted[-50:]
 
print('TOP POS: ', top_pos)
print('TOP NEG: ', top_neg)

