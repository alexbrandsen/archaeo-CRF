import sklearn

import sklearn_crfsuite

from sklearn.externals import joblib

import word2features


# open CRF model from file 
filename = 'crf-v1.sav'
crf = joblib.load(filename)


# input should be like [['RAAP-RAPPORT', 'SPEC'], ['2217', 'TW']]	
def predictSentence(possedTokenList):
	x = [word2features.sent2features(s) for s in possedTokenList]
	
	output = crf.predict(x)[0] # x = list of lists of dicts
	#output = crf.predict_single(x)[0] # x = (list of dicts) – feature dicts in python-crfsuite format (for single sentence) 
	return output

#array = [[['RAAP-RAPPORT', 'SPEC'], ['2217', 'TW']]]
#output = predictSentence(array)
#print(output)
