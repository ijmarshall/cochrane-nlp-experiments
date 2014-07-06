# Runs a multitask model for sentences
# Models all domains together using SVMs with L2 regularisation + bag of words
# Uses feature limiting to exclude features with < 2 appearances across the corpus

# 1 fold, and versus humans and baseline
# uses bigrams and unigrams

from cochranenlp.experiments import riskofbias
from cochranenlp.ml import modhashvec
from cochranenlp.output import metrics, outputnames

from itertools import izip

import numpy as np

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier

import sys
import os
import time


def main(out_dir="results"):
    model_metrics = metrics.BinaryMetricsRecorder(domains=riskofbias.CORE_DOMAINS)
    stupid_metrics = metrics.BinaryMetricsRecorder(domains=riskofbias.CORE_DOMAINS)


    # parse the risk of bias data from Cochrane
    data = riskofbias.RoBData(test_mode=False)
    data.generate_data(doc_level_only=False)

    docs = riskofbias.MultiTaskSentFilter(data)

    uids = np.array(docs.get_ids())
    no_studies = len(uids)

    kf = KFold(no_studies, n_folds=5, shuffle=False)

    tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
    

    for train, test in kf:

        X_train_d, y_train, i_train = docs.Xyi(uids[train])

        interactions = {domain:[] for domain in riskofbias.CORE_DOMAINS}
        for doc_domain in i_train:
            for domain in riskofbias.CORE_DOMAINS:
                if domain == doc_domain:
                    interactions[domain].append(True)
                else:
                    interactions[domain].append(False)

        vec = modhashvec.ModularVectorizer(norm=None, non_negative=True, binary=True, ngram_range=(1, 2), n_features=2**26) # since multitask + bigrams = huge feature space
        vec.builder_clear()

        

        # import pdb; pdb.set_trace()

    
        vec.builder_add_docs(X_train_d, low=10) # add base features

        for domain in riskofbias.CORE_DOMAINS:
            X_train_d = docs.X_iter(uids[train])
            print np.sum(interactions[domain]), "/", len(interactions[domain]), "added for", domain
            vec.builder_add_interaction_features(X_train_d, interactions=interactions[domain], prefix=domain+"-i-", low=2) # then add interactions

    
        X_train = vec.builder_fit_transform()
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='accuracy')
        clf.fit(X_train, y_train)

        # free some memory now, only need the model
        del X_train_d # remove references to these
        del X_train

        del y_train
        clf = clf.best_estimator_ # and we only need the best performing, discard the rest

        # Test on each domain in turn

        filtered_data = riskofbias.SentFilter(data)

        for domain in riskofbias.CORE_DOMAINS:


            X_test_d, y_test = filtered_data.Xy(uids[test], domain=domain)

            
            # build up test vector

            vec.builder_clear()
            vec.builder_add_docs(X_test_d) # add base features
            vec.builder_add_docs(X_test_d, prefix=domain+'-i-') # add interactions

            X_test = vec.builder_transform()

            y_preds = clf.predict(X_test)

            model_metrics.add_preds_test(y_preds, y_test, domain=domain)
            stupid_metrics.add_preds_test([1] * len(y_test), y_test, domain=domain)

            del X_test_d, X_test, y_test, y_preds




    model_metrics.save_csv(os.path.join(out_dir, outputnames.filename(label="model")))
    stupid_metrics.save_csv(os.path.join(out_dir, outputnames.filename(label="stupid-baseline")))
    




if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        print "output directory: %s" % args[1]
        main(out_dir=args[1])
    else:
        main()