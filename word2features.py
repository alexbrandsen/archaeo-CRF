import csv

# Function to convert a csv file to a list of dictionaries.  Takes in one variable called "variables_file"
def getTermsFromCsv(variables_file):
	# Open variable-based csv, iterate over the rows and map values to a list of dictionaries containing key/value pairs
	reader = csv.DictReader(open(variables_file, 'r'))
	output = []
	for line in reader:
		for columnName in line:
			if (columnName[0:7] == 'concept' or columnName[0:8] == 'altLabel') and len(line[columnName]) > 0:
				output.append(line[columnName].lower())
	return output
	
materials = getTermsFromCsv('../ontologies/abr/ABR-materialen.csv')
dates = getTermsFromCsv('../ontologies/abr/ABR-periodes.csv')
artefacts = getTermsFromCsv('../ontologies/abr/ABR-artefacten.csv')

#print(artefacts)
#exit(0)

def word2features(sent, i):
	#print (sent[i])
	word = sent[i][0]
	postag = sent[i][1]
	
	# "+ngram:existsIn" means the current word is the start word of a phrase in the ontology 
	# "-ngram:existsIn" means the current word is the 2nd or 3rd word of a phrase in the ontology


	# IDEAS to add to features
	# - add previous two predicted labels
	# - word2vec (clusters)
	# - fasttext (clusters)
	# conjunction of the previous tag and the current token (http://www.aclweb.org/anthology/W15-1830)
	# prefix / suffix

	features = {
		'bias': 1.0,
		'word.lower()': word.lower(),
		'word.isupper()': word.isupper(),
		'word.istitle()': word.istitle(),
		'word.isdigit()': word.isdigit(),
		'postag': postag,
		'+ngram:existsInMaterials': True if word.lower() in materials else False,
		'+ngram:existsInDates': True if word.lower() in dates else False,
		'+ngram:existsInArtefacts': True if word.lower() in artefacts else False,
	}
	
	if i > 0:
		wordMin1 = sent[i-1][0]
		postagMin1 = sent[i-1][1]
		features.update({
			'-1:word.lower()': wordMin1.lower(),
			'-1:word.istitle()': wordMin1.istitle(), # first char uppercase, rest lowercase
			'-1:word.isupper()': wordMin1.isupper(),
			'-1:word.isdigit()': wordMin1.isdigit(),
			'-1:postag': postagMin1,
			'-1:existsInMaterials': True if wordMin1.lower() in materials else False,
			'-1:existsInDates': True if wordMin1.lower() in dates else False,
			'-1:existsInArtefacts': True if wordMin1.lower() in artefacts else False,
			'-ngram:existsInMaterials': True if wordMin1.lower()+' '+word.lower() in materials else False,
			'-ngram:existsInDates': True if wordMin1.lower()+' '+word.lower() in dates else False,
			'-ngram:existsInArtefacts': True if wordMin1.lower()+' '+word.lower() in artefacts else False,
		})
	else:
		features['BOS'] = True
	
	if i < len(sent)-1:
		#print sent[i+1]
		wordPlus1 = sent[i+1][0]
		postagPlus1 = sent[i+1][1]
		features.update({
			'+1:word.lower()': wordPlus1.lower(),
			'+1:word.istitle()': wordPlus1.istitle(),
			'+1:word.isupper()': wordPlus1.isupper(),
			'+1:word.isdigit()': wordPlus1.isdigit(),
			'+1:postag': postagPlus1,
			'+1:existsInMaterials': True if wordPlus1.lower() in materials else False,
			'+1:existsInDates': True if wordPlus1.lower() in dates else False,
			'+1:existsInArtefacts': True if wordPlus1.lower() in artefacts else False,
			'+ngram:existsInMaterials': True if word.lower()+' '+wordPlus1.lower() in materials or features['+ngram:existsInMaterials'] else False,
			'+ngram:existsInDates': True if word.lower()+' '+wordPlus1.lower() in dates or features['+ngram:existsInDates'] is True else False,
			'+ngram:existsInArtefacts': True if word.lower()+' '+wordPlus1.lower() in artefacts or features['+ngram:existsInArtefacts'] is True else False,
		})
	else:
		features['EOS'] = True
	
	# trigrams
	if i > 1:
		wordMin2 = sent[i-2][0]	
		postagMin2 = sent[i-2][1]	
		features.update({
			'-2:word.lower()': wordMin2.lower(),
			'-2:word.istitle()': wordMin2.istitle(),
			'-2:word.isupper()': wordMin2.isupper(),
			'-2:word.isdigit()': wordMin2.isdigit(),
			'-2:postag': postagMin2,
			'-2:existsInMaterials': True if wordMin2.lower() in materials else False,
			'-2:existsInDates': True if wordMin2.lower() in dates else False,
			'-2:existsInArtefacts': True if wordMin2.lower() in artefacts else False,
			'-ngram:existsInMaterials': True if wordMin2.lower()+' '+wordMin1.lower()+' '+word.lower() in materials or features['-ngram:existsInMaterials'] is True else False,
			'-ngram:existsInDates': True if wordMin2.lower()+' '+wordMin1.lower()+' '+word.lower() in dates or features['-ngram:existsInDates'] is True else False,
			'-ngram:existsInArtefacts': True if wordMin2.lower()+' '+wordMin1.lower()+' '+word.lower() in artefacts or features['+ngram:existsInArtefacts'] is True else False,
		})
	if i > 0 and i < len(sent)-1:
		features.update({
			'-ngram:existsInMaterials': True if wordMin1.lower()+' '+word.lower()+' '+wordPlus1.lower() in materials or features['-ngram:existsInMaterials'] is True else False,
			'-ngram:existsInDates': True if wordMin1.lower()+' '+word.lower()+' '+wordPlus1.lower() in dates or features['-ngram:existsInDates'] is True else False,
			'-ngram:existsInArtefacts': True if wordMin1.lower()+' '+word.lower()+' '+wordPlus1.lower() in artefacts or features['-ngram:existsInArtefacts'] is True else False,
		})
	if i < len(sent)-2:
		wordPlus2 = sent[i+2][0]	
		postagPlus2 = sent[i+2][1]	
		features.update({
			'+2:word.lower()': wordPlus2.lower(),
			'+2:word.istitle()': wordPlus2.istitle(),
			'+2:word.isupper()': wordPlus2.isupper(),
			'+2:word.isdigit()': wordPlus2.isdigit(),
			'+2:postag': postagPlus2,
			'+2:existsInMaterials': True if wordPlus2.lower() in materials else False,
			'+2:existsInDates': True if wordPlus2.lower() in dates else False,
			'+2:existsInArtefacts': True if wordPlus2.lower() in artefacts else False,
			'+ngram:existsInMaterials': True if word.lower()+' '+wordPlus1.lower()+' '+wordPlus2.lower() in materials or features['+ngram:existsInMaterials'] is True else False,
			'+ngram:existsInDates': True if word.lower()+' '+wordPlus1.lower()+' '+wordPlus2.lower() in dates or features['+ngram:existsInDates'] is True else False,
			'+ngram:existsInArtefacts': True if word.lower()+' '+wordPlus1.lower()+' '+wordPlus2.lower() in artefacts or features['+ngram:existsInArtefacts'] is True else False,
		})
				
	#if word.lower() in dates:
		#print(features)
		#print(features['+ngram:existsInDates'])
		#exit(0)

	return features


def sent2features(sent):
	return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	return [label for token, postag, label in sent]

def sent2tokens(sent):
	return [token for token, postag, label in sent]
