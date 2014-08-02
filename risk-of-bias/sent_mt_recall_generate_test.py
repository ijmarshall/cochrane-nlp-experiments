# Runs a multitask model for sentences
# Models all domains together using SVMs with L2 regularisation + bag of words
# Uses feature limiting to exclude features with < 2 appearances across the corpus
# with CLASS WEIGHTING **(note that we're therefore weighting across all domains together)**
# 1 fold, and versus humans and baseline
# uses bigrams and unigrams

from cochranenlp.experiments import riskofbias2 as riskofbias
from cochranenlp.ml import modhashvec2 as modhashvec
from cochranenlp.output import metrics, outputnames

from random import randrange

from itertools import izip

import numpy as np

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier

import os
import time

import unicodecsv as csv


def main():

    model_metrics = metrics.BinaryMetricsRecorder(domains=riskofbias.CORE_DOMAINS)
    stupid_metrics = metrics.BinaryMetricsRecorder(domains=riskofbias.CORE_DOMAINS)

    f = open('test_data.csv','wb')
    w = csv.DictWriter(f, ["pmid", "domain", "sent_text", "random", "human", "algorithm", "top3", "top1"], escapechar="\\")
    w.writeheader()

    # parse the risk of bias data from Cochrane     
    data = riskofbias.RoBData(test_mode=False)
    data.generate_data(doc_level_only=False)

    docs = riskofbias.MultiTaskSentFilter(data)

    uids = np.array(docs.get_ids())
    no_studies = len(uids)

    kf = KFold(no_studies, n_folds=5, shuffle=False)

    tuned_parameters = {"alpha": np.logspace(-4, -1, 5), "class_weight": [{1: i, -1: 1} for i in np.logspace(0, 2, 5)]}

    vec = modhashvec.ModularVectorizer(norm=None, non_negative=True, binary=True, ngram_range=(1, 2), n_features=2**26) # since multitask + bigrams = huge feature space

    for k_i, (train, test) in enumerate(kf):

        if k_i == 1:
            break

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




            y_df = clf.decision_function(X_test) # get distances from the decision boundary
            # positive distances = more likely to be relevant sentences

            r_len = len(y_preds)
            y_top3 = []
            y_top1 = []
            y_rand = []

            y_uids = np.array(docs.y_uids(uids[test], domain=domain))

            # import pdb; pdb.set_trace()

            for y_uid in np.unique(y_uids):

                mask = np.where(y_uids == y_uid)[0]
                doc_df = y_df[mask]

                doc_top3 = np.argpartition(doc_df, -3)[-3:]
                y_top3.extend(list(mask[doc_top3]))
                
                doc_top1 = np.argmax(doc_df)
                y_top1.append(mask[doc_top1])

                doc_rand = np.random.randint(0, len(doc_df))
                y_rand.append(mask[doc_rand])


            human_sent_indices = np.where(y_test==1)[0]
            algorithm_sent_indices = np.where(y_preds==1)[0]

            model_metrics.add_preds_test(y_preds, y_test, domain=domain)
            stupid_metrics.add_preds_test([-1] * len(y_test), y_test, domain=domain)

            # import pdb; pdb.set_trace()

            for doc_i, (doc, pmid) in enumerate(izip(docs.X(uids[test], domain=domain), docs.iter_pmid(uids[test], domain=domain))):

                row = {"domain": domain,
                       "sent_text": doc,
                       "random": doc_i in y_rand,
                       "human": doc_i in human_sent_indices,
                       "algorithm": doc_i in algorithm_sent_indices,
                       "top3": doc_i in y_top3,
                       "top1": doc_i in y_top1,
                       "pmid": pmid}

                if row["random"] or row["human"] or row["top3"] or row["top1"]:
                    # please note, the sentences will only be included in the analysis if
                    # in the top1 or top3
                    # we do have data on whether the raw classifier has predicted yes/no
                    # 
                    # this in effect means where the classifier picks <= 3 sentences
                    # we use all raw classifier data
                    # where >3 sentences are predicted by raw classifier, only the
                    # top 3 are used; the rest are discarded
                    w.writerow(row)

            del X_test, y_test, y_preds

        del clf



    model_metrics.save_csv(os.path.join('results', outputnames.filename(label="model")))
    stupid_metrics.save_csv(os.path.join('results', outputnames.filename(label="stupid-baseline")))
    f.close()
    




if __name__ == '__main__':
    main()