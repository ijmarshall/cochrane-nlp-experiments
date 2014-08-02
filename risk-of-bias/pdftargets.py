
#
#	PDFs to obtain
#

import csv
from cochranenlp.experiments import riskofbias2 as riskofbias
from cochranenlp.readers import biviewer
from cochranenlp.output.progressbar import ProgressBar

import yaml



from collections import Counter, defaultdict
import codecs

import cochranenlp
import re

import os

REGEX_QUOTE = re.compile("\"(.*?)\"") # retrive blocks of text in quotes

domain_list = riskofbias.CORE_DOMAINS + ["UNKNOWN DOMAIN"]

def _load_domain_map(filename=os.path.join(cochranenlp.PATH, "data", "domain_names.txt")):

    with codecs.open(filename, 'rb', 'utf-8') as f:
        raw_data = yaml.load(f)

    mapping = {}
    for key, value in raw_data.iteritems():
        for synonym in value:
            mapping[synonym] = key

    return mapping



data = riskofbias.RoBData(test_mode=False)
data.generate_data(doc_level_only=False)

mapper = _load_domain_map()

all_pmids = Counter()
domains_present = defaultdict(Counter)


b = biviewer.BiViewer()
print "getting all pubmed ids in CDSR..."

p = ProgressBar(len(b))

for doc in b:
	p.tap()
	all_pmids[doc.pubmed['pmid']] += 1
	quality_items = doc.cochrane["QUALITY"]
	
	for quality_item in quality_items:
		try:
			mapped_domain = mapper[quality_item['DOMAIN']]
		except KeyError:
			mapped_domain = "UNKNOWN DOMAIN"


		if mapped_domain in domain_list: # we don't want known but out of category domains
			domains_present[doc.pubmed['pmid']][mapped_domain] += 1


		# if quality_item['DESCRIPTION']:
		# 	quotes = REGEX_QUOTE.findall(quality_item['DESCRIPTION'])

		# 	if quotes:
		# 		domains_present[doc.pubmed['pmid']][mapped_domain] += 1


multi_assessed = [pmid for pmid, count in all_pmids.iteritems() if count >1]

print "Total multi-assessed = %d" % len(multi_assessed)


docs = riskofbias.DocFilter(data)
uids = docs.get_ids() # what we have PDFs for

print "PDFs we have = %d" % len(uids)

to_get = set(multi_assessed).difference(uids)

print "no to get = %d" % len(to_get)


# output = {uid: {domain: False for domain in riskofbias.CORE_DOMAINS} for uid in uids}

# for domain in riskofbias.CORE_DOMAINS:
# 	domain_uids = docs.get_ids(filter_domain=domain)
# 	for domain_uid in domain_uids:
# 		output[domain_uid][domain] = True

    
    
flip_output = []

possibly_relevant_to_get = list(set(multi_assessed).difference(uids))
# get rid of ones we have already


for uid in possibly_relevant_to_get:

	if all((domains_present[uid][domain] < 2 for domain in domain_list)):
		continue
		# skip if all ratings < 2
	row = domains_present[uid].copy() # avoid messing around with the original (though probably fine..)
	row["pubmed id"] = uid
	flip_output.append(row)
		
		# for uid, results in output.iteritems():
		# 	row = results
		# 	row["pubmed id"] = uid
		# 	flip_output.append(row)
		


with open("to_get.csv", "wb") as f:
	w = csv.DictWriter(f, ["pubmed id"] + domain_list)
	w.writeheader()
	w.writerows(flip_output)

	
