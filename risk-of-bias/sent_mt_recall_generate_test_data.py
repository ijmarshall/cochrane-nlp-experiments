# Runs a multitask model for sentences
# Models all domains together using SVMs with L2 regularisation + bag of words
# Uses feature limiting to exclude features with < 2 appearances across the corpus

# 1 fold, and versus humans and baseline
# uses bigrams and unigrams

from cochranenlp.experiments import riskofbias2 as riskofbias
from cochranenlp.ml import modhashvec2 as modhashvec
from cochranenlp.output import metrics, outputnames

from itertools import izip

import numpy as np

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier

import os
import time


def main():

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
    
    vec = modhashvec.ModularVectorizer(norm=None, non_negative=True, binary=True, ngram_range=(1, 2), n_features=2**26) # since multitask + bigrams = huge feature space

    for train, test in kf:

        y_train = docs.y(uids[train])

            
        vec.builder_clear()
        vec.builder_add_interaction_features(docs.X(uids[train]), low=7) # add base features
        vec.builder_add_interaction_features(docs.X_i(uids[train]), low=2) # then add interactions
        X_train = vec.builder_fit_transform()

        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall')

        # import pdb; pdb.set_trace()

        clf.fit(X_train, y_train)
        del X_train, y_train
        clf = clf.best_estimator_ # and we only need the best performing, discard the rest

        # Test on each domain in turn

        # filtered_data = riskofbias.SentFilter(data)

        for domain in riskofbias.CORE_DOMAINS:

            print "Testing on %s" % domain

            vec.builder_clear()
            vec.builder_add_interaction_features(docs.X(uids[test], domain=domain)) # add base features
            vec.builder_add_interaction_features(docs.X_i(uids[test], domain=domain)) # then add interactions
            X_test = vec.builder_transform()

            y_test = docs.y(uids[test], domain=domain)
            y_preds = clf.predict(X_test)

            model_metrics.add_preds_test(y_preds, y_test, domain=domain)
            stupid_metrics.add_preds_test([-1] * len(y_test), y_test, domain=domain)

            del X_test, y_test, y_preds

        del clf



    model_metrics.save_csv(os.path.join('results', outputnames.filename(label="model")))
    stupid_metrics.save_csv(os.path.join('results', outputnames.filename(label="stupid-baseline")))
    




if __name__ == '__main__':
    main()