'''
This outputs training data for the risk of bias task -- 
specifically, document text and label pairs -- for 
consumption by downstream approaches. Will also 
output corresponding train/test id sets. 
'''

import pdb 
import csv 

import numpy as np

import cochranenlp
from cochranenlp.experiments import riskofbias

def _dump_Xy(docs, lbls, ids, outpath): 
    assert (len(docs) == len(lbls))
    with open(outpath, 'wb') as outf: 
        csv_writer = csv.writer(outf)
        for doc, lbl, id_ in zip(docs, lbls, ids): 
            csv_writer.writerow([doc, str(lbl)])


def generate_all_training_data():

    data = riskofbias.RoBData(test_mode=False)
    data.generate_data(doc_level_only=True)
    filtered_data = riskofbias.DocFilter(data)

    uids_all = filtered_data.get_ids(pmid_instance=0) # those with 1 or more assessment (i.e. all)
    uids_double_assessed = filtered_data.get_ids(pmid_instance=1) # those with 2 (or more) assessments (to hide for training)
    uids_train = np.setdiff1d(uids_all, uids_double_assessed)

    uids_test = {}
    for domain in riskofbias.CORE_DOMAINS:
        X_train_d, y_train, domain_uids_train = filtered_data.Xy(uids_train, domain=domain, 
                                                            pmid_instance=0, 
                                                            return_uids=True)

        domain_str = domain.replace(" ", "-")
        # dump training ids per 
        with open("train-uids-%s.txt" % domain_str, 'wb') as outf: 
            csv_writer = csv.writer(outf)     
            csv_writer.writerow(domain_uids_train)


        # and dump the actual data!
        _dump_Xy(X_train_d, y_train, "train-Xy-%s.txt" % domain_str)

        # get domain test ids
        # (i.e. the double assessed trials, which have a judgement for the current domain in
        #   *both* the 0th and 1st review)
        uids_domain_all = filtered_data.get_ids(pmid_instance=0, filter_domain=domain)
        uids_domain_double_assessed = filtered_data.get_ids(pmid_instance=1, filter_domain=domain)
        uids_test_domain = np.intersect1d(uids_domain_all, uids_domain_double_assessed)

        X_test_d, y_test, uids_test = filtered_data.Xy(uids_test_domain, domain=domain, 
                                                        pmid_instance=0, return_uids=True)
        # so uids_test should, i think, match uids_test_domain here!
        assert(all(uids_test == uids_test_domain))
        with open("test-uids-%s.txt" % domain_str, 'wb') as outf: 
            csv_writer = csv.writer(outf)       
            csv_writer.writerow(uids_test)

        _dump_Xy(X_test_d, y_test, "test-Xy-%s.txt" % domain_str)

        X_ignore, y_human = filtered_data.Xy(uids_test_domain, domain=domain, pmid_instance=1)
        X_ignore = None # don't need this bit


### @TODO finish
### ??? one file per doc id? 

### i think actually one big file makes more sense,
### containing all labels and data
#
#   doc_id, doc_lbl, sentence_num, sentence, sentence_lbl
#
###
def _dump_sentence_X_ys(doc_ids, doc_lbls, sentences, sent_lbls, outpath):
    assert(len(doc_ids) == len(sentences))
    with open(outpath, 'wb') as outf: 
        csv_writer = csv.writer(outf)
        csv_writer.writerow(
            ["doc_id", "doc_lbl", "sentence_number", "sentence", "sentence_lbl"])
        for doc_idx, id_ in enumerate(doc_ids): 
            doc_sent_lbls = sent_lbls[doc_idx]
            for sent_idx, sent in enumerate(sentences[doc_idx]):
                csv_writer.writerow(
                  [str(id_), str(doc_lbls[doc_idx]), str(sent_idx), sent, str(doc_sent_lbls[sent_idx])])
    

def generate_all_training_data_w_sentences():

    data = riskofbias.RoBData(test_mode=False) # switch flag to false...
    data.generate_data(doc_level_only=False, skip_small_files=True)

    filtered_data = riskofbias.DocFilter(data)
    
    uids_all = filtered_data.get_ids(pmid_instance=0) # those with 1 or more assessment (i.e. all)
    uids_double_assessed = filtered_data.get_ids(pmid_instance=1) # those with 2 (or more) assessments (to hide for training)
    uids_train = np.setdiff1d(uids_all, uids_double_assessed)

    sent_docs = riskofbias.SentFilter(data)

    uids_test = {}
    for domain in riskofbias.CORE_DOMAINS:

        sent_uids = np.intersect1d(uids_train, np.array(sent_docs.get_ids(filter_domain=domain)))
        

        X_train_sents, y_train_sents = sent_docs.Xy(sent_uids, domain=domain, split_by_doc=True)
        X_train_d, y_train_d, domain_uids_train = filtered_data.Xy(uids_train, domain=domain, 
                                                            pmid_instance=0, 
                                                            return_uids=True)
        #X_train_d, y_train, uids_train = filtered_data.Xy(uids_train, domain=domain, 
        #                                                    pmid_instance=0, 
        #                                                    return_doc_uids=True)
        
        #import pdb; pdb.set_trace()

        domain_str = domain.replace(" ", "-")
        # dump training ids per; redundant with below but what the hell
        with open("train-uids-w-sentences-%s.txt" % domain_str, 'wb') as outf: 
            csv_writer = csv.writer(outf)     
            csv_writer.writerow(domain_uids_train)


        # and dump the actual data!
        #_dump_Xy(X_train_d, y_train, domain_uids_train, "train-Xy-%s.txt" % domain_str)
        #doc_ids, doc_lbls, sentences, sent_lbls, outpath
        _dump_sentence_X_ys(domain_uids_train, y_train_d, X_train_sents, 
                                y_train_sents, "train-Xy-w-sentences-%s.txt" % domain_str)

        # get domain test ids
        # (i.e. the double assessed trials, which have a judgement for the current domain in
        #   *both* the 0th and 1st review)
        uids_domain_all = filtered_data.get_ids(pmid_instance=0, filter_domain=domain)
        uids_domain_double_assessed = filtered_data.get_ids(pmid_instance=1, filter_domain=domain)
        uids_test_domain = np.intersect1d(uids_domain_all, uids_domain_double_assessed)


        test_sent_uids = np.intersect1d(uids_test_domain, np.array(sent_docs.get_ids(filter_domain=domain)))
        X_test_sents, y_test_sents = sent_docs.Xy(test_sent_uids, domain=domain, split_by_doc=True)
        X_test_d, y_test_d, uids_test_domain = filtered_data.Xy(uids_test_domain, domain=domain, 
                                                        pmid_instance=0, return_uids=True)
       
        with open("test-uids-w-sentences-%s.txt" % domain_str, 'wb') as outf: 
            csv_writer = csv.writer(outf)       
            csv_writer.writerow(uids_test_domain)

        #_dump_Xy(X_test_d, y_test, "test-Xy-%s.txt" % domain_str)
        _dump_sentence_X_ys(uids_test_domain, y_test_d, X_test_sents, 
                                y_test_sents, "test-Xy-w-sentences-%s.txt" % domain_str)


        X_ignore, y_human = filtered_data.Xy(uids_test_domain, domain=domain, pmid_instance=1)
        X_ignore = None # don't need this bit

   
