from __future__ import print_function
import pdb 
import csv
import sys  
csv.field_size_limit(sys.maxsize)

import os
import time
import datetime
import data_helpers
import theano
import numpy as np
import nltk 
import gensim
from gensim.models import Word2Vec
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import accuracy_score

# pointer to Kim's repo
sys.path.append("CNN_sentence")
import conv_net_sentence 

np.random.seed(1337)  # for reproducibility


def read_RoB_data(path="train-Xy-Random-sequence-generation.txt", 
                    fit_vectorizer=True, y_tuples=False, zero_one=True,
                    max_features=50000): 
    ''' 
    note that this method is agnostic as to whether the 
    data is train/test 
    '''
    raw_texts, X, y = [], [], []
    with open(path) as input_file: 
        rows = csv.reader(input_file)
        for row in rows: 
            doc_text, lbl = row
            raw_texts.append(doc_text)
            cur_y = int(lbl)
            if y_tuples:
                if cur_y > 0:
                    y.append(np.array([0,1]))
                else: 
                    y.append(np.array([1,0]))
            else:
                if cur_y < 1:
                    if zero_one:
                        y.append(0)
                    else:
                        y.append(-1)
                    #pdb.set_trace()
                    
                else:
                    y.append(1)

    if fit_vectorizer:
        vectorizer = CountVectorizer(ngram_range=(1,1), binary=True, 
                                        max_features=max_features)

        vectorizer.fit(raw_texts)
        return raw_texts, y, vectorizer
        
    return raw_texts, y 


def to_token_indices(docs, v): 
    tokenized = []
    vocab_size = len(v.vocabulary_)
    for d in docs: 
        tokenized_doc = []
        for t in nltk.word_tokenize(d): 
            t = t.lower()
            if t not in v.vocabulary_: continue
            try:
                cur_idx = v.vocabulary_[t]    #### vocabulary_  maps terms to indices
            except: 
                # this will serve as our `unk' token
                #cur_idx = vocab_size
                pass 
            tokenized_doc.append(cur_idx+1)    #######Note that we index word from 1, leave index 0 for zero padding
        tokenized.append(tokenized_doc)
    return tokenized


def load_trained_w2v_model(path="PubMed-w2v.bin"):

    m = Word2Vec.load_word2vec_format(path, binary=True)
    return m

    '''
    word_vecs = {}
    with open(path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
               f.read(binary_len)
    return word_vecs
    '''

def _get_init_vectors(vectorizer, wv, unknown_words_to_vecs=None):
    if unknown_words_to_vecs is None: 
        unknown_words_to_vecs = {}
    init_vectors = []
    #for token_idx, t in enumerate(vectorizer.vocabulary_):   ###key is word, value is index
    for t in vectorizer.vocabulary_:
        t = t.lower()
        try:
            init_vectors.append(wv[t])
        except:
            if not t in unknown_words_to_vecs:      ###words in the dataset but not in word embedding
                # initialize randomly; is this really the 
                # best option?
                v = np.random.uniform(-0.25,0.25,wv.vector_size)
                unknown_words_to_vecs[t] = v 
            init_vectors.append(unknown_words_to_vecs[t])
    #init_vectors = np.vstack(init_vectors)
    init_vectors = np.array(init_vectors)
    return init_vectors, unknown_words_to_vecs


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
        Pad each sequence to the same length:
        the length of the longest sequence.
        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
        the end of the sequence.
        Supports post-padding and pre-padding (default).
        Parameters:
        -----------
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
        Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue # empty list was found
        if len(s)>=maxlen:
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % padding)
            x[idx,:] = trunc
        else:
            if padding == 'post':
                x[idx,:len(s)] = s
                x[idx,len(s):] = 0
            elif padding == 'pre':
                x[idx,-len(s):] = s
                x[idx,:len(s)] = 0
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
    return x

# @TMP limiting to 2000 words!
def RoB_CNN_theano(maxlen=2000):
    '''
    Process data for CNN classification via the 
    theano implementation (Kim, 2014)
    '''
    
    ###
    # read in data (this also fits a vectorizer for us)
    train_docs, y_train, vectorizer = read_RoB_data(path="train-Xy-Random-sequence-generation.txt", 
                                        y_tuples=False, max_features=50000)
    vocab_size = len(vectorizer.vocabulary_)
    print ("vocabulary size of training data: " + str(vocab_size))
    test_docs, y_test = read_RoB_data(path="test-Xy-Random-sequence-generation.txt",
                                        fit_vectorizer=False,
                                        y_tuples=False)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # map to word indices
    X_train = to_token_indices(train_docs, vectorizer)  ##convert raw texts to word indices
    X_test  = to_token_indices(test_docs, vectorizer)

    # read in pretrained word vectors
    wv = load_trained_w2v_model()
    wv_dim = wv.vector_size
    print ("dimension of word embedding: " + str(wv_dim))
    # set initial word vectors for all token indices
    init_vectors, unk_vecs = _get_init_vectors(vectorizer, wv)
    W = init_vectors.astype(theano.config.floatX)
    W = np.vstack((np.zeros(wv_dim),W))       #####add zeros vectors as the 0th word embedding
    W = W.astype(theano.config.floatX)
    print ("dimension of W matrix is: " + str(W.shape))    #####should be 50001 times 200
    # zero-pad sentences
    ## @TMP only one set of filters
    #filter_heights = [3,4,5]
    filter_heights = [3,4,5]
    pad_len = maxlen + 2*(max(filter_heights)-1)
    X_train = pad_sequences(X_train, maxlen=maxlen, padding="post", dtype=np.int32) ##X_train is indices
    X_test  = pad_sequences(X_test, maxlen=maxlen, padding="post", dtype=np.int32)
    print('X_train shape: ', X_train.shape)
    print('X_test shape: ', X_test.shape)

    ''' 
    Kim's code expects datasets to contain train and test 
    matrices, *with the labels as the last entries!*
    '''
    X_y_train = np.array(np.hstack((X_train, np.matrix(y_train).T)))
    X_y_test  = np.array(np.hstack((X_test, np.matrix(y_test).T)))
    datasets = [X_y_train, X_y_test]

    # @TMP setting these to small values 
    n_filters  = 100 # number of feature maps per height
    batch_size = 50
    n_epochs   = 20
    perf = conv_net_sentence.train_conv_net(datasets,
                      W,
                      img_w = W.shape[1],
                      lr_decay=0.95,
                      filter_hs=filter_heights,
                      conv_non_linear="relu",
                      hidden_units=[n_filters,2], 
                      shuffle_batch=True, 
                      n_epochs=n_epochs, 
                      sqr_norm_lim=9,
                      non_static=True,
                      batch_size=batch_size, 
                      dropout_rate=[0.5])
    
    return perf 


if  __name__ =='__main__': 
    perf = RoB_CNN_theano()
    print("perf: %s" % perf)
    with open("perf", 'w') as outf:
        outf.write(str(perf))
