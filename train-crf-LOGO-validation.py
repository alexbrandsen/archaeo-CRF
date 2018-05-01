# ------------ imports --------------

#from itertools import chain

import os

import numpy

import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import LeaveOneGroupOut

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from sklearn.externals import joblib

from collections import Counter

import pprint
pp = pprint.PrettyPrinter(indent=4)

import word2features

import time
start_time = time.time()

# ------------ functions --------------

def file2list(fileLocation):
	outputList = []
	global currentGroup
	currentGroup += 1
	with open(fileLocation, 'r') as myfile:
		sentences = myfile.read().split('\n\n')
		for sentence in sentences:
			sentenceList = []
			words = sentence.split('\n')
			for word in words:
				if word:
					wordsList = []
					attributes = word.split(' ')
					for attribute in attributes:
						# take out classes we don't need
						if attribute in tagsToTakeOut:
							attribute = 'O'
						wordsList.append(attribute)
					sentenceList.append(wordsList)
			outputList.append(sentenceList)
			groups.append(currentGroup)
	
	return outputList


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

# ------------ start script --------------

# leave out some tags for now
tagsToTakeOut = ('B-ACT', 'I-ACT','B-CON', 'I-CON','B-EXC', 'I-EXC','B-MON', 'I-MON','B-RES', 'I-RES', 'B-LOC', 'I-LOC')

groups = []
currentGroup = 0


train_sents = []
folder = 'individual-docs'
for filename in os.listdir(folder):
	train_sents += file2list(folder+'/'+filename)#.pop()

#train_sents.pop()

X = [word2features.sent2features(s) for s in train_sents]
y = [word2features.sent2labels(s) for s in train_sents]

# convert to numpy array for logo
X = numpy.array(X)
y = numpy.array(y)

logo = LeaveOneGroupOut()

overall_y_pred = numpy.array([])
overall_y_test = numpy.array([])

for train_index, test_index in logo.split(X, y, groups):
	
	X_train = X[train_index]
	X_test = X[test_index]
	y_train = y[train_index]
	y_test = y[test_index]

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
	
	overall_y_pred = numpy.concatenate((overall_y_pred, y_pred))
	overall_y_test = numpy.concatenate((overall_y_test, y_test))
	
	numpy.save('overall_y_pred.npy', overall_y_pred)
	numpy.save('overall_y_test.npy', overall_y_test)
	
	# change I and B to just the entity, to get better understandable f1 etc
	for i in range(0,len(overall_y_pred)): 
		for j in range(0,len(overall_y_pred[i])): 
			if overall_y_pred[i][j] is not 'O':
				overall_y_pred[i][j] = overall_y_pred[i][j].replace('B-','')
				overall_y_pred[i][j] = overall_y_pred[i][j].replace('I-','')	
	for i in range(0,len(overall_y_test)): 
		for j in range(0,len(overall_y_test[i])): 
			if overall_y_test[i][j] is not 'O':
				overall_y_test[i][j] = overall_y_test[i][j].replace('B-','')
				overall_y_test[i][j] = overall_y_test[i][j].replace('I-','')	
			
	#print(overall_y_pred, overall_y_test)
	#exit()

			
sorted_labels = labels
# group B and I results (not needed if B- and I- taken out)
#sorted_labels = sorted(
#	labels,
#	key=lambda name: (name[1:], name[0])
#)


print(str(len(Counter(groups).keys()))+'-fold leave one group out cross validation results:')
print(metrics.flat_classification_report(
	overall_y_test, overall_y_pred, labels=sorted_labels, digits=3
))

print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))

print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-20:])


print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])

print ('Took '+ str(time.time() - start_time) + ' seconds')
#exit(0)

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



print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))

print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-20:])


print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])


# save CRF model to file (open again with "crf = joblib.load(filename)")
filename = 'crf-v1.sav'
joblib.dump(crf, filename)

print ('Took '+ str(time.time() - start_time) + ' seconds')

