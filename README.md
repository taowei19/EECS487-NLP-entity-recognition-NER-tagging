# EECS487-NLP-entity-recognition-NER-tagging
train a Hidden Markov Model (HMM) to do named entity recognition (NER) tagging:
Steps:
1.train an HMM from given corpus
2.use Beam Search to find the most likely hidden state sequence, find the best smoothing k and beam_width, the final correctness will be 93% (for 2000 sentences).
3.use the Forward Algorithm to calculate the probability of a specific sequence of observations without knowledge of the hidden states.(Another algorithm, based on the pervious code and further calculation. Similar to viteribi algorithm.)


About Named Entity Recognition¶
Named Entity Recognition (NER) is the task of identifying "named entities", which are typically proper nouns associated with people (e.g. "Oprah Winfrey"), places (e.g. "Scotland"), organizations (e.g. "Ford Motor Company"), and more.

NER tagging refers to the process of assigning a tag to each word in a sentence indicating (1) whether or not it is a named entitity and (2), if it is a named entity, what type of named entity it is.

The dataset we will be using has several possible tags:

"O" : not a named entity
"B-PER" / "I-PER" : a person
"B-LOC" / "I-LOC" : a location
"B-ORG" / "I-ORG" : an organization
"B-MISC" / "I-MISC" : a miscellaneous other type of named entity
To properly tag multi-word named entities, each NER tag has either a "B-" or an "I-" prepended to it. The "B-" tags refer to the first word in a (possibly) multi-word named entity. The "I-" tags refer to the subsequent words in the same named entity.

So, for instance, the sentence "Tom Cruise flew to Hawaii on Delta Airlines" would be tagged as follows:

["B-PER", "I-PER", "O", "O", "B-LOC", "O", "B-ORG", "I-ORG"]

If you need to see the code part, pls email me. taowe@umich.edu
