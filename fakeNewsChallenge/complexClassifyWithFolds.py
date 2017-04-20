




import nltk
#from nltk.corpus import names
import csv
import re

def mapStanceToNumber(stance):
	if stance == "agree":
		return 3
	elif stance == "discuss":
		return 2
	elif stance == "disagree":
		return 1
	else:
		return 0

def clean(s):
	return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def wordLength_features(word):
	return {'word_length': len(word)}

def avgWordLength_features(text):
	words = text.split()
	return {'avg_word_len' : sum(len(x) for x in words)/len(words)}

def numWords_features(text):
	words = text.split()
	return {'num_words' : len(words)}

def matchedWords_features(text1,text2):
	global ignoreWords
	matches = 0
	swords = text1.split()
	twords = text2.split()
	for oneWord in swords:
		if oneWord in twords and oneWord not in ignoreWords:
			matches += 1
	return {'words_matched' : matches}

def totalMatchedWords_features(text1,text2):
	global ignoreWords
	matches = 0
	swords = text1.split()
	twords = text2.split()
	for oneWord in swords:
	 if oneWord not in ignoreWords:
		matches += twords.count(oneWord)
	return {'tot_matches' : matches}

def matchNgram_features(text1,text2,num):
	from nltk.util import ngrams
	matches = 0
	token = nltk.word_tokenize(text1)
	n_grams = ngrams(token,num)
	n_gram_strings = [' '.join(t) for t in n_grams]
	for t in n_gram_strings:
		if t in text2:
			matches += 1
	return {str(num)+'_gram_matches' : matches}

#def matchNouns_features(text1,text2):
#def matchVerbs_features(text1,text2):
def matchPOS_features(text1,text2):
	global ignoreWords
	nouns = 0
	verbs = 0
	posTag1 = nltk.pos_tag(nltk.word_tokenize(text1))
	#posTag2 = nltk.pos_tag(nltk.word_tokenize(text2))
	nouns1 = [word for (word,pos) in posTag1 if pos.startswith('NN')]
	#nouns2 = [word for (word,pos) in posTag2 if pos.startswith('NN')]
	verbs1 = [word for (word,pos) in posTag1 if pos.startswith('VB')]
	#verbs2 = [word for (word,pos) in posTag2 if pos.startswith('VB')]
	for sNoun in nouns1:
		if sNoun in text2 and sNoun not in ignoreWords:
			nouns += 1
	for sVerb in verbs1:
		if sVerb in text2 and sVerb not in ignoreWords:
			verbs += 1
	#print "Nouns are ",nouns," verbs are ",verbs
	return {'noun_matches' : nouns,
		'verb_matches' : verbs}

def matchInversions_features(text1,text2):
	headMatch = 0
	bodyMatch = 0
	invertingWords = ['fake','fraud','hoax','false','deny','denies','despite','nope','doubt','doubts','bogus','debunk','pranks','retract']
#,'not'
	token1 = text1.split()
	token2 = text2.split()
	for word in invertingWords:
		if word in token1:
			headMatch += token1.count(word)
		if word in token2:
			bodyMatch += token2.count(word)
	return {'head_invert_matches' : headMatch,
		'body_invert_matches' : bodyMatch}

def matchBogus_features(text1,text2):
	headMatch = 0
	bodyMatch = 0
	bogusWords = ['boom','fantastic','trump','pizzagate','epic fail']
	token1 = text1.split()
	token2 = text2.split()
	for word in bogusWords:
		if word in token1:
			headMatch += token1.count(word)
		if word in token2:
			bodyMatch += token2.count(word)
	return {'head_bogus_matches' : headMatch,
		'body_bogus_matches' : bodyMatch}

def matchStems_features(text1,text2):
	import nltk.tokenize.punkt
	import nltk.stem.snowball
	import string
	tokenizer = nltk.tokenize.punkt.PunktWordTokenizer()
	stemmer = nltk.stem.snowball.SnowballStemmer('english')

	tokens_a = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(text1) if token.lower().strip(string.punctuation) not in ignoreWords]
	tokens_b = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(text2) if token.lower().strip(string.punctuation) not in ignoreWords]
	stems_a = [stemmer.stem(token) for token in tokens_a]
	stems_b = [stemmer.stem(token) for token in tokens_b]
	return {'stems_matches' : len(set(stems_a).intersection(stems_b)) / float(len(set(stems_a).union(stems_b))) }


ignoreWords = nltk.corpus.stopwords.words('english')
print "There are ",len(ignoreWords),"Ignore Words."
#ignoreWords = ['is','at','of','with','in','as','on','and','to','be','a']
#print "Ignore Words are: ",ignoreWords

def compareText_features(text1,text2):
	result = {}
	result.update(matchNgram_features(text1,text2,1))
	result.update(matchNgram_features(text1,text2,2))
	result.update(matchNgram_features(text1,text2,3))
	result.update(matchNgram_features(text1,text2,4))
	result.update(matchNgram_features(text1,text2,5))
	#result.update(matchBigram_features(text1,text2))
	result.update(matchPOS_features(text1,text2))
	result.update(matchedWords_features(text1,text2))
	result.update(totalMatchedWords_features(text1,text2))
	result.update(matchInversions_features(text1,text2))
	result.update(matchBogus_features(text1,text2))
	#result.update(matchStems_features(text1,text2))
	#print "result: ",result
	return result

print "Opening files..."
with open('train_bodies.csv', 'r') as csvBodyFile:
	rbody = csv.reader(csvBodyFile, delimiter=',', quotechar='"')
	body = list(rbody)

with open('train_stances.csv', 'r') as csvHeaderFile:
	rheaders = csv.reader(csvHeaderFile, delimiter=',', quotechar='"')
	headers = list(rheaders)

print "Files read in."

articles = []
## headline, body, agreement
for row in headers:
	for brow in body:
		if (row[1] == brow[0]):
			##articles += [(row[0],brow[1],row[2]),]
			articles += [(clean(row[0]),clean(brow[1]),row[2]),]
			#print "Headline: ",row[0]," Agreement: ",row[2]
			#print "Matched Words: ",matchedWords_features(row[0],brow[1])
			#print "Tot Matches: ",totalMatchedWords_features(row[0],brow[1])
			#compareText_features(row[0],brow[1])
			#matchBigram_features(row[0],brow[1])
			#print matchStems_features(row[0],brow[1])
			#print ""
			#print "WordLengthAvg: ",numWords_features(row[0])
			#print "WordLengthAvg: ",numWords_features(brow[1])

print "Finished combining articles; filter and randomize"
print len(articles)

fairDistribArticles = []
#capacity = 250
##capacity = 50
capacity = 800

train_size = capacity * 1 # There are 4 capacity sets, half equals total/2
test_size = (capacity * 4) - train_size  # Get what's left

curCapacity = [0,0,0,0]
for story in articles:
	numStance = mapStanceToNumber(story[2])
	if curCapacity[numStance] < capacity:
		curCapacity[numStance] += 1
		fairDistribArticles += [(story)]

print "Finished filter/Radomizing"	
print len(fairDistribArticles)
articles = fairDistribArticles

import random
random.shuffle(articles)

nbSum = 0
dtSum = 0
num_folds = 2 
subset_size = len(articles)/num_folds

for i in range(num_folds):
	featuresets = [(compareText_features(n,m), agreement) for (n, m, agreement) in articles]
	#featuresets = [(totalMatchedWords_features(n,m), agreement) for (n, m, agreement) in articles]
	#train_set, test_set = featuresets[train_size:], featuresets[:test_size]
	train_set = featuresets[i*subset_size:][:subset_size]
	test_set = featuresets[:i*subset_size] + featuresets[(i+1)*subset_size:]
	classifier = nltk.NaiveBayesClassifier.train(train_set)

#import pickle
#f = open('my_classifier.pickle','wb')
#pickle.dump(classifier,f)
#f.close()

##print "Name is ",name1,classifier.classify(verb_features(name1))
#for ref in range(501,540):
#	print "Article is ",articles[ref][0]," ",articles[ref][1][0:100]," ",articles[ref][2]," ",classifier.classify(totalMatchedWords_features(articles[ref][0], articles[ref][1]))

	print sorted(classifier.labels())
	#classifier.classify_many(test)

	nbAccuracy = nltk.classify.accuracy(classifier, test_set)
	print( "naive bayes: ",nbAccuracy, )

#for pdist in classifier.prob_classify(test_set):
#	print('%.4f $.4f' % (pdist.prob('disagree'),pdist.prob('agree')))

	##classifier.show_most_informative_features(20)


#errors = []
#for (headline,body,stance) in articles:
#	guess = classifier.classify(totalMatchedWords_features(headline,body))
#	if guess != stance:
#		errors.append( (stance, guess, headline, body) )

#for (stance, guess, headline, body) in sorted(errors):
#	print ('correct={0:<8} guess={1:<8} headline={2:<30} body={3:<50}'.format(stance,guess, headline, body[:200]) )


	classifier = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)

	print sorted(classifier.labels())
	dtAccuracy = nltk.classify.accuracy(classifier, test_set)
	print( "dec tree:",dtAccuracy )

	#print(classifier)
	nbSum += nbAccuracy
	dtSum += dtAccuracy

print "********"
print "k-fold accruacy, k=",num_folds
print "Naive   Bayes: ",nbSum/num_folds
print "Decision Tree: ",dtSum/num_folds
