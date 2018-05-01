
#from itertools import chain

import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from sklearn.externals import joblib

import pprint
pp = pprint.PrettyPrinter(indent=4)

import word2features

import time
start_time = time.time()

# leave out some tags for now
tagsToTakeOut = ('B-ACT', 'I-ACT','B-CON', 'I-CON','B-EXC', 'I-EXC','B-MON', 'I-MON','B-RES', 'I-RES')

def file2list(fileLocation):
	outputList = []
	with open(fileLocation, 'r') as myfile:
		sentences = myfile.read().split('\n\n')
		for sentence in sentences:
			sentenceList = []
			words = sentence.split('\n')
			for word in words:
				wordsList = []
				attributes = word.split(' ')
				for attribute in attributes:
					# take out classes we don't need
					if attribute in tagsToTakeOut:
						attribute = 'O'
					wordsList.append(attribute)
				sentenceList.append(wordsList)
			outputList.append(sentenceList)
	
	return outputList
	

train_sents = file2list("train-and-test/ned.train")
test_sents = file2list("train-and-test/ned.testb")

#print train_sents[0]

# remove empty element at end of file due to last line break
train_sents.pop()
test_sents.pop()


#print sent2features(train_sents[0])[0]

X_train = [word2features.sent2features(s) for s in train_sents]
y_train = [word2features.sent2labels(s) for s in train_sents]

X_test = [word2features.sent2features(s) for s in test_sents]
y_test = [word2features.sent2labels(s) for s in test_sents]

#pp.pprint(X_train[1])
#exit(0)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

labels = list(crf.classes_)
labels.remove('O')

#print labels


y_pred = crf.predict(X_test)

#f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
#print(f1)


# group B and I results
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

# save CRF model to file (open again with "crf = joblib.load(filename)")
#filename = 'finalized_model.sav'
#joblib.dump(crf, filename)

#exit(0)

# define fixed parameters and parameters to search
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

# use the same metric for evaluation
f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)

# search
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
rs.fit(X_train, y_train)


# crf = rs.best_estimator_
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))


crf = rs.best_estimator_
y_pred = crf.predict(X_test)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))


from collections import Counter

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))

print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-20:])


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])


# save CRF model to file (open again with "crf = joblib.load(filename)")
filename = 'crf-v1.sav'
joblib.dump(crf, filename)

print ('Took '+ str(time.time() - start_time) + ' seconds')

