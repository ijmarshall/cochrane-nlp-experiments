
import unicodecsv as csv
from cochranenlp.experiments import riskofbias
import json
import uuid
import datetime
from collections import defaultdict
import random

DOMAINS = riskofbias.CORE_DOMAINS[:6]
NO_QUESTIONS = 2 # number of studies to ask about
INPUT = "test_data.csv"

DOMAIN_DESCRIPTIONS = {}


INTRODUCTION = """
<b>Thank you for helping with our RobotReviewer validation study!</b>

We are going to present you with a series of sentences from a number of studies.

Please note, <b>it doesn't matter whether the text indicates a low, high, or unknown risk of bias</b>.
We are only interested in whether the sentence contains information which is relevant to making a decision.

This is not a test of your expertise. We expect the relevance (or not) of each sentence to be quickly obvious most of the time.
"""

DOMAIN_DESCRIPTIONS[DOMAINS[0]] = """
Random sequence generation: potentially biased allocation to interventions due to inadequate generation of a randomised sequence

There is a low risk of bias if the investigators describe a random sequence 
generation process such as: referring to a random number table, using a computer random number 
generator, coin tossing, shuffling cards or envelopes, throwing dice, drawing of lots.

There is a high risk of selection bias if the investigators describe a non-random sequence generation process, such as: sequence generated by odd or even date of birth, date (or day) of 
admission, hospital or clinic record number; or allocation by judgement of the clinician, preference of 
the participant, results of a laboratory test or a series of tests, or availability of the intervention. 
"""


DOMAIN_DESCRIPTIONS[DOMAINS[1]] = """
Allocation concealment (selection bias): Biased allocation to interventions due to inadequate concealment of allocations prior to assignment 

There is a low risk of selection bias if the participants and investigators enrolling participants could not 
foresee assignment because one of the following, or an equivalent method, was used to conceal 
allocation: central allocation (including telephone, web-based and pharmacy-controlled randomization); 
sequentially numbered drug containers of identical appearance; or sequentially numbered, opaque, 
sealed envelopes. 

There is a high risk of bias if participants or investigators enrolling participants could possibly foresee 
assignments and thus introduce selection bias, such as allocation based on: using an open random 
allocation schedule (e.g. a list of random numbers); assignment envelopes were used without 
appropriate safeguards (e.g. if envelopes were unsealed or non-opaque or not sequentially numbered); 
alternation or rotation; date of birth; case record number; or other explicitly unconcealed procedures. 
"""


DOMAIN_DESCRIPTIONS[DOMAINS[2]] = """
Blinding of participants: Bias due to knowledge of the allocated interventions by participants during the study

There is a low risk of performance bias if blinding of participants was ensured and it was unlikely that the 
blinding could have been broken; or if there was no blinding or incomplete blinding, but the review 
authors judge that the outcome is not likely to be influenced by lack of blinding.

Blinding of personnel/care providers: Bias due to knowledge of the allocated interventions by personnel/care providers during the study. 

There is a low risk of performance bias if blinding of personnel was ensured and it was unlikely that the 
blinding could have been broken; or if there was no blinding or incomplete blinding, but the review 
authors judge that the outcome is not likely to be influenced by lack of blinding
"""


DOMAIN_DESCRIPTIONS[DOMAINS[3]] = """
Blinding of outcome assessor: Bias due to knowledge of the allocated interventions by outcome assessor

There is low risk of detection bias if the blinding of the outcome assessment was ensured and it was 
unlikely that the blinding could have been broken; or if there was no blinding or incomplete blinding, but 
the review authors judge that the outcome is not likely to be influenced by lack of blinding.; or: 
- for patient-reported outcomes in which the patient was the outcome assessor (e.g., pain, disability): 
there is a low risk of bias for outcome assessors if there is a low risk of bias for participant blinding.* 
- for outcome criteria that are clinical or therapeutic events that will be determined by the interaction 
between patients and care providers (e.g., co-interventions, length of hospitalization, treatment 
failure), in which the care provider is the outcome assessor: there is a low risk of bias for outcome 
assessors if there is a low risk of bias for care providers.* 
- for outcome criteria that are assessed from data from medical forms: there is a low risk of bias if the 
treatment or adverse effects of the treatment could not be noticed in the extracted data.
"""

DOMAIN_DESCRIPTIONS[DOMAINS[4]] = """
Incomplete outcome data (attrition bias): Attrition bias due to amount, nature or handling of incomplete outcome data

There is a low risk of attrition bias if there were no missing outcome data; reasons for missing outcome 
data were unlikely to be related to the true outcome (for survival data, censoring unlikely to be 
introducing bias); missing outcome data were balanced in numbers, with similar reasons for missing 
data across groups.

For dichotomous outcome data, the proportion of missing outcomes compared 
with the observed event risk was not enough to have a clinically relevant impact on the intervention 
effect estimate; for continuous outcome data, the plausible effect size (difference in means or 
standardized difference in means) among missing outcomes was not enough to have a clinically relevant 
impact on observed effect size, or missing data were imputed using appropriate methods

"""

DOMAIN_DESCRIPTIONS[DOMAINS[5]] = """
Selective Reporting: reporting bias due to selective outcome reporting

Criteria for a judgement of 'Low risk' of bias.
Any of the following:
The study protocol is available and all of the study's pre-specified (primary and secondary) outcomes that are of interest in the review have been reported in the pre-specified way;
The study protocol is not available but it is clear that the published reports include all expected outcomes, including those that were pre-specified (convincing text of this nature may be uncommon).

Criteria for the judgement of 'High risk' of bias.
Any one of the following:
Not all of the study's pre-specified primary outcomes have been reported;
One or more primary outcomes is reported using measurements, analysis methods or subsets of the data (e.g. subscales) that were not pre-specified;
One or more reported primary outcomes were not pre-specified (unless clear justification for their reporting is provided, such as an unexpected adverse effect);
One or more outcomes of interest in the review are reported incompletely so that they cannot be entered in a meta-analysis;
The study report fails to include results for a key outcome that would be expected to have been reported for such a study.


"""



def main():

	question_data = {domain: defaultdict(list) for domain in DOMAINS}

	# parse data
	with open(INPUT, 'rb') as f:
		reader = csv.DictReader(f)

		for i, row in enumerate(reader):
			row_dom = row.pop("domain")
			if row_dom in DOMAINS:
				pmid = row.pop('pmid')
				row["sent_text"] = "[question %d] <em>%s</em>" % (i, row["sent_text"].strip())
				question_data[row_dom][pmid].append(row)

	# generate questions

	output = []

	output.append("::NewPage:: Welcome")
	output.append(INTRODUCTION)


	for domain in DOMAINS:

		output.append("::NewPage:: %s - reminder" % domain)

		output.append("Here's a reminder of what <b>%s</b> comprises; once you're happy, click the Next button below to start the evaluation." % domain)
		output.append("Remember, we are asking you to judge whether pieces of text are relevant to a risk of bias domain; <em>not</em> whether they indicate high or low bias.")

		output.append("<em>" + DOMAIN_DESCRIPTIONS[domain] + "</em>")

	

		test_ids = random.sample(question_data[domain].keys(), NO_QUESTIONS)

		for i, uid in enumerate(test_ids):

			output.append("::NewPage:: %s - study %d/%d" % (domain, int(i)+1, NO_QUESTIONS))
			output.append("How relevant are each of the following sentences to the domain <b>%s</b>?" % domain)

			for j, row in enumerate(question_data[domain][uid]):


				

				output.append("""
%s
() Highly relevant
() Some relevance
() Not relevant

""" % row['sent_text'].strip())

			# print "study number %s " % i
			# print "\n".join(l['sent_text'].strip() for l in question_data[domain][i])
			# print



	with open("output.txt", "wb") as f:
		f.write("\n\n".join(output))



if __name__ == '__main__':
	main()
