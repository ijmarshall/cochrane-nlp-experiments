# Runs a basic model
# Models each domain in turn using SVMs with L2 regularisation + bag of words
# Uses feature limiting to exclude features with < 2 appearances across the corpus

# Keeps double assessed studies as withheld (since will use later as a test set)

from cochranenlp.experiments import riskofbias
import numpy as np

def main():

    # parse the risk of bias data from Cochrane
    data = riskofbias.RoBData(test_mode=True)
    data.generate_data(doc_level_only=False)

    # filter the data by Document
    filtered_data = riskofbias.DocFilter(data)

    # get the uids of the desired training set
    # (for this experiment those which appear in only one review)

    uids_all = filtered_data.get_ids(pmid_instance=0) # those with 1 or more assessment (i.e. all)
    uids_double_assessed = filtered_data.get_ids(pmid_instance=1) # those with 2 (or more) assessments (to hide for training)

    uids_train = np.setdiff1d(uids_all, uids_double_assessed)

    # we need different test ids for each domain
    # (since we're testing on studies with more than one RoB assessment for *each domain*)

    uids_test = {}
    for domain in riskofbias.CORE_DOMAINS:
        uids_domain_all = filtered_data.get_ids(pmid_instance=0, filter_domain=domain)
        uids_domain_double_assessed = filtered_data.get_ids(pmid_instance=1, filter_domain=domain)
        uids_test[domain] = np.intersect1d(uids_domain_all, uids_domain_multi_assessed)


    print uids_train

    experiment = riskofbias.ExperimentBase()

    experiment.run(filtered_data, uids_train)



    






if __name__ == '__main__':
    main()