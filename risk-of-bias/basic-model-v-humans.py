# Runs a basic model
# Models each domain in turn using SVMs with L2 regularisation + bag of words
# Uses feature limiting to exclude features with < 2 appearances across the corpus

# 1 fold, and versus humans and baseline


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
    dat = riskofbias.RoBData(test_mode=True)
    dat.generate_data(doc_level_only=True)


    model_metrics = metrics.BinaryMetricsRecorder(domains=riskofbias.CORE_DOMAINS)

    stupid_metrics = metrics.BinaryMetricsRecorder(domains=riskofbias.CORE_DOMAINS)


    multitask_docs = riskofbias.MultiTaskDocFilter(dat) # use the same ids as the multitask model
    multitask_uids = np.array(multitask_docs.get_ids())
    no_studies = len(multitask_uids)


    kf = KFold(no_studies, n_folds=5, shuffle=False)

    for domain in riskofbias.CORE_DOMAINS:

        docs = riskofbias.DocFilter(dat)
        uids = np.array(docs.get_ids(filter_domain=domain))

        print "%d docs obtained for domain: %s" % (len(uids), domain)


        tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='f1')

        no_studies = len(uids)


        for train, test in kf:

            X_train_d, y_train = docs.Xy(np.intersect1d(uids, multitask_uids[train]), domain=domain)
            X_test_d, y_test = docs.Xy(np.intersect1d(uids, multitask_uids[test]), domain=domain)

            vec = modhashvec.InteractionHashingVectorizer(norm=None, non_negative=True, binary=True)

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