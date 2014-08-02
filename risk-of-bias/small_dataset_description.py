# Runs a basic model
# Models each domain in turn using SVMs with L2 regularisation + bag of words
# Uses feature limiting to exclude features with < 2 appearances across the corpus

# 5 fold cross validation


from cochranenlp.experiments import riskofbias
from cochranenlp.ml import modhashvec
from cochranenlp.output import metrics, outputnames


import numpy as np

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier

import os
import time





def main():
    dat = riskofbias.RoBData(test_mode=False)
    dat.generate_data(doc_level_only=False)



    

    

    docs = riskofbias.DocFilter(dat)

    sents = riskofbias.SentFilter(dat)

    for domain in riskofbias.CORE_DOMAINS:
        print domain
        print "="*40
        print


        uids = np.array(docs.get_ids(filter_domain=domain))
        X, y = docs.Xy(uids, domain=domain)


        print "%d/%d (=%.2f) are positive" % (np.sum(np.array(y)==1), len(y), (float(np.sum(np.array(y)==1)) * 100/float(len(y))))

        sent_uids = np.array(sents.get_ids(filter_domain=domain))
        sent_X, sent_y = sents.Xy(uids, domain=domain)

        print "%d/%d (=%.2f) are positive" % (np.sum(np.array(sent_y)==1), len(sent_y), (float(np.sum(np.array(sent_y)==1)) * 100/float(len(sent_y))))
        print
        print




if __name__ == '__main__':
    main()