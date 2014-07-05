# Runs a basic sentence model
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
    dat.generate_data(doc_level_only=True)

    

    model_metrics = metrics.BinaryMetricsRecorder(domains=riskofbias.CORE_DOMAINS)
    stupid_metrics = metrics.BinaryMetricsRecorder(domains=riskofbias.CORE_DOMAINS)

    docs = riskofbias.SentFilter(dat)


    for domain in riskofbias.CORE_DOMAINS:

        uids = np.array(docs.get_ids(filter_domain=domain))
        no_studies = len(uids)

        kf = KFold(no_studies, n_folds=5, shuffle=False)

        print "%d docs obtained for domain: %s" % (no_studies, domain)

        tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}

        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall')

        

        for train, test in kf:

            X_train_d, y_train = docs.Xy(uids[train], domain=domain)
            X_test_d, y_test = docs.Xy(uids[test], domain=domain)

            vec = modhashvec.InteractionHashingVectorizer(norm=None, non_negative=True, binary=True, ngram_range=(1, 2), n_features=2**24)

            X_train = vec.fit_transform(X_train_d, low=2)
            X_test = vec.transform(X_test_d)

            clf.fit(X_train, y_train)

            y_preds = clf.predict(X_test)

            model_metrics.add_preds_test(y_preds, y_test, domain=domain)

            stupid_metrics.add_preds_test([1] * len(y_test), y_test, domain=domain)

    model_metrics.save_csv(os.path.join('results', outputnames.filename(label="model")))
    stupid_metrics.save_csv(os.path.join('results', outputnames.filename(label="stupid-baseline")))


if __name__ == '__main__':
    main()