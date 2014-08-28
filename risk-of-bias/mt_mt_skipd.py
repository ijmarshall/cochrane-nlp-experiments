# Multitask Multitask Model

# 1 fold, and versus humans and baseline
# uses bigrams and unigrams

# SKIPS THE SELECTIVE REPORTING DOMAIN
# since this is likely introducing noise into the multitask model


from cochranenlp.experiments import riskofbias2 as riskofbias
from cochranenlp.ml import modhashvec2 as modhashvec
from cochranenlp.output import metrics, outputnames

from cochranenlp.textprocessing.tokenizer import sent_tokenizer

import numpy as np

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier

import sys
import os
import time
import pdb
from itertools import izip

skip_domains=riskofbias.CORE_DOMAINS[:5] + riskofbias.CORE_DOMAINS[6:] # skip domain 5

def main(out_dir="results"):
    model_metrics = metrics.BinaryMetricsRecorder(domains=skip_domains)
    stupid_metrics = metrics.BinaryMetricsRecorder(domains=skip_domains)
    human_metrics = metrics.BinaryMetricsRecorder(domains=skip_domains)

    # parse the risk of bias data from Cochrane
    print "risk of bias data!"
    data = riskofbias.RoBData(test_mode=False)
    data.generate_data(doc_level_only=False)

    # filter the data by Document
    filtered_data = riskofbias.DocFilter(data)

    # get the uids of the desired training set
    # (for this experiment those which appear in only one review)

    uids_all = filtered_data.get_ids(pmid_instance=0) # those with 1 or more assessment (i.e. all)
    uids_double_assessed = filtered_data.get_ids(pmid_instance=1) # those with 2 (or more) assessments (to hide for training)
    uids_train = np.setdiff1d(uids_all, uids_double_assessed)



    ########################
    # sentence prediction  #
    ########################


    # The first stage is to make the sentence prediction model using the
    #   training data set
    #
    print "First, making sentence prediction model"
    sent_docs = riskofbias.MultiTaskSentFilter(data)
    uids = np.array(sent_docs.get_ids(domain=skip_domains))
    no_studies = len(uids)

    # sentence tokenization
    sent_vec = modhashvec.ModularVectorizer(norm=None, non_negative=True, binary=True, ngram_range=(1, 2), n_features=2**26) # since multitask + bigrams = huge feature space
    sent_vec.builder_clear()
    # add base features; this effectively generates the shared feature
    # space (i.e., features for all domains)
    sent_vec.builder_add_interaction_features(sent_docs.X(uids_train, domain=skip_domains), low=7) 

    # now we add interaction features, which cross the domain with the
    # tokens. specifically, the X_i method returns token tuples crossing 
    # every term with every domain, and the vectorizer (an instance of 
    # ModularVectorizer) deals with inserting the actual interaction tokens 
    # that cross domains with tokens.
    domain_interaction_tuples = sent_docs.X_i(uids_train, domain=skip_domains)
    sent_vec.builder_add_interaction_features(domain_interaction_tuples, low=2) 

    # setup sentence classifier
    tuned_parameters = {"alpha": np.logspace(-4, -1, 5), "class_weight": [{1: i, -1: 1} for i in np.logspace(0, 2, 5)]}
    # bcw: are we sure we want to do 'recall' here, and not (e.g.) F1?
    sent_clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall', n_jobs=4)

    X_train = sent_vec.builder_fit_transform()
    y_train = sent_docs.y(uids_train, domain=skip_domains)

    sent_clf.fit(X_train, y_train)
    del X_train, y_train
    # we only need the best performing
    sent_clf = sent_clf.best_estimator_ 

    # now we have our multi-task sentence prediction model,
    # which we'll use to make sentence-level predictions for
    # documents.


    ########################
    # document prediction  #
    ########################

    # we need different test ids for each domain
    # (since we're testing on studies with more than one RoB assessment for *each domain*)
    docs = riskofbias.MultiTaskDocFilter(data)
    X_train_d = docs.Xyi(uids_train, domain=skip_domains)


    tuned_parameters = {"alpha": np.logspace(-4, -1, 5), "class_weight": [{1: i, -1: 1} for i in np.logspace(-1, 1, 5)]}
    clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall', n_jobs=4)

    # bcw: note that I've amended the y method to 
    # return interactions as well (i.e., domain strs)
    y_train = docs.y(uids_train, domain=skip_domains)

    # add interaction features (here both domain + high prob sentences)
    interactions = {domain:[] for domain in skip_domains}
    high_prob_sents = []
    interaction_domains = []

    for doc_index, (doc_text, doc_domain) in enumerate(X_train_d):
        
        doc_sents = sent_tokenizer.tokenize(doc_text)
        doc_domains = [doc_domain] * len(doc_sents)
        # interactions
        doc_X_i = izip(doc_sents, doc_domains)

        # sent_vec is from above.
        sent_vec.builder_clear()
        sent_vec.builder_add_interaction_features(doc_sents) # add base features
        sent_vec.builder_add_interaction_features(doc_X_i) # then add interactions
        doc_sents_X = sent_vec.builder_transform()

        ## bcw -- shouldn't we use the *true* sentence labels
        # here, rather than predictions????

        # sent_clf was trained above
        doc_sents_preds = sent_clf.predict(doc_sents_X)

        high_prob_sents.append(" ".join([sent for sent, sent_pred in 
                        zip(doc_sents, doc_sents_preds) if sent_pred==1]))
        interaction_domains.append("-s-" + doc_domain)

        if doc_index % 10 == 0:
            print doc_index
        # from collections import Counter
        # prob_count = Counter(list(doc_sents_preds))
        # print prob_count
        
        # for domain in riskofbias.CORE_DOMAINS:
        #     if domain == doc_domain:
        #         interactions[domain].append(True)
        #     else:
        #         interactions[domain].append(False)

    vec = modhashvec.ModularVectorizer(norm=None, non_negative=True, binary=True, ngram_range=(1, 2), n_features=2**26) # since multitask + bigrams = huge feature space
    vec.builder_clear()
    vec.builder_add_docs(docs.X(uids_train, domain=skip_domains), low=7) # add base features
    vec.builder_add_docs(docs.Xyi(uids_train, domain=skip_domains), low=2) # add domain interactions
    # removed X_train_d since already been through the generator! (needed reset)
    vec.builder_add_docs(izip(high_prob_sents, interaction_domains), low=2)    # then add sentence interaction terms

    X_train = vec.builder_fit_transform()    
    clf.fit(X_train, y_train)

    ############
    # testing  #
    ############

    # Test on each domain in turn
    for domain in skip_domains:
        uids_domain_all = filtered_data.get_ids(pmid_instance=0, filter_domain=domain)
        uids_domain_double_assessed = filtered_data.get_ids(pmid_instance=1, filter_domain=domain)
        uids_test_domain = np.intersect1d(uids_domain_all, uids_domain_double_assessed)

        X_test_d, y_test = filtered_data.Xy(uids_test_domain, domain=domain, pmid_instance=0)
        X_ignore, y_human = filtered_data.Xy(uids_test_domain, domain=domain, pmid_instance=1)
        X_ignore = None # don't need this bit

        #
        #   get high prob sents from test data
        #
        high_prob_sents = []
        
        for doc_text in X_test_d:
            doc_sents = sent_tokenizer.tokenize(doc_text)

            # bcw -- I think this (using doc_domain and not 
            # domain) was the bug before!
            #doc_domains = [doc_domain] * len(doc_sents)
            doc_domains = [domain] * len(doc_sents)

            doc_X_i = izip(doc_sents, doc_domains)

            sent_vec.builder_clear()
            sent_vec.builder_add_interaction_features(doc_sents) # add base features
            sent_vec.builder_add_interaction_features(doc_X_i) # then add interactions
            doc_sents_X = sent_vec.builder_transform()
            doc_sents_preds = sent_clf.predict(doc_sents_X)

            high_prob_sents.append(" ".join(
                                    [sent for sent, sent_pred in 
                                        zip(doc_sents, doc_sents_preds) if sent_pred==1]))

        
        sent_domain_interactions = ["-s-" + domain] * len(high_prob_sents)
        domain_interactions = [domain] * len(high_prob_sents)

        print
        print "domain: %s" % domain
        print "High prob sents:"
        print '\n'.join(high_prob_sents)

        # build up test vector
        vec.builder_clear()
        vec.builder_add_docs(X_test_d) # add base features
        vec.builder_add_docs(izip(X_test_d, domain_interactions)) # add interactions
        vec.builder_add_docs(izip(high_prob_sents, sent_domain_interactions)) # sentence interactions
    
        X_test = vec.builder_transform()
        y_preds = clf.predict(X_test)

        model_metrics.add_preds_test(y_preds, y_test, domain=domain)
        human_metrics.add_preds_test(y_human, y_test, domain=domain)
        stupid_metrics.add_preds_test([1] * len(y_test), y_test, domain=domain)


    model_metrics.save_csv(os.path.join(out_dir, outputnames.filename(label="model")))
    stupid_metrics.save_csv(os.path.join(out_dir, outputnames.filename(label="stupid-baseline")))
    human_metrics.save_csv(os.path.join(out_dir, outputnames.filename(label="human-performance")))   


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        print "output directory: %s" % args[1]
        main(out_dir=args[1])
    else:
        main()
