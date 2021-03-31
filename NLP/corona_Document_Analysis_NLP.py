# Knowledge Extraction of COVID-19 articles 
# Python SpaCy
# Regular Expressions
# NLP

import pandas as pd
import os
import spacy
import numpy as np
import seaborn as sb
import re
from spacy.matcher import Matcher
from spacy.tokens import Doc
from numpy import array
from sklearn.feature_extraction.text import CountVectorizer # Vectorizer for cbow
from collections import Counter
from matplotlib import pyplot as plt


#from timer import  Timer
import time
#Use bag of to get gather words that are useful for medical care

nlp = spacy.load("en_core_web_sm")

# Search items
pattern1 = [{"LOWER": "medical"}]
pattern2 = [{"LOWER": "care"}]
pattern3 = [{"LOWER": "diagnosis"}]
pattern4 = [{"LOWER": "treatment"}]
pattern5 = [{"LOWER": "therapy"}]
pattern5 = [{"LOWER": "effective"}]
pattern6 = [{"LOWER": "medical care"}]
pattern7 = [{"LOWER": "vaccine"}]
pattern8 = [{"LOWER": "vaccines"}]
pattern9 = [{"LOWER": "drug"}]
pattern10 = [{"LOWER": "vitamin"}]
pattern11 = [{"LOWER": "nursing"}]
pattern12 = [{"LOWER": "medical staff"}]
pattern13 = [{"LOWER": "Acute Respiratory Distress Syndrome"}] # not found in the text
pattern14 = [{"LOWER": "Extracorporeal membrane oxygenation"}]
pattern15 = [{"LOWER": "ventilation"}]
pattern16 = [{"LOWER": "manifestations"}]
pattern17 = [{"LOWER": "EUA"}]
pattern18 = [{"LOWER": "CLIA"}]
pattern19 = [{"LOWER": "elastomeric"}]
pattern20 = [{"LOWER": "N95"}]
pattern21 = [{"LOWER": "telemedicine"}]
pattern22 = [{"LOWER": "outcomes"}]

# NLP lists and dictionary
word_list = []
word_dict = {}
allWords = []

text_list = [['', '']]
medical_care = [['', '', '']]

#customize_stop_words = [
#    'From','from', 'To', 'to', 'Hospital', 'hospital', '-', ')', '(', ',', ':', 'of', 'for', 'the', 'The', 'is',
#	'[', ']', ';', "\xa0", '/', 'virus', 'studies', '1', 'BACKGROUND', 'population', 'previously', 'countries', 'dogs', 'data',
#	'infection', '%', 'viral'
#]

customize_stop_words = [
     "\xa0", ','
]
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True

os.chdir("/home/saul/corona/CORD-19-research-challenge/2020-03-13")
filename = 'all_sources_metadata_2020-03-13.csv'


def readfile():

	textcount = 0
	print("reading the file")
	coronafile = pd.read_csv(filename, sep=',')

	#check dup rows
	duprows = coronafile[coronafile.duplicated(keep = False)]
	print(len(duprows))
	#print(duprows)
	#save duplicate rows to csv
	duprows.to_csv("duplicates.csv")
	
	#remove duplicated rows
	nodupcorona = coronafile.drop_duplicates(subset=None, keep='first', inplace=False)
	print(len(nodupcorona))

	#check source_x
	#print(nodupcorona.source_x.unique()) #['CZI' 'PMC' 'biorxiv' 'medrxiv']

	#check doi
	#print(nodupcorona.doi.unique())

	#authors
	#print(nodupcorona.authors.value_counts())

	#describe authors
	#print(nodupcorona.authors.describe()) # top authors ['Ehrt, Christiane', 'Brinkjost, Tobias', 'Koc.

	# check all columns
	#print("Licence ", nodupcorona.describe(include = 'all'))

	#'WHO #Covidence'
	#print(nodupcorona['WHO #Covidence'].value_counts())

	#drop not useful variables

	drop_list = ["WHO #Covidence"]

	nodupcorona = nodupcorona.drop(drop_list, axis=1)
	#print(nodupcorona.describe(include='all'))
	#print("Columns ", nodupcorona.columns)

	analyseAbstract(nodupcorona.sha, nodupcorona.abstract, textcount)
	print ("Text Count", textcount)
	#check data
	#sampledata = nodupcorona[nodupcorona.sha == '0a00a6df208e068e7aa369fb94641434ea0e6070']
	#print(sampledata.abstract)

def analyseAbstract(sha, abstract, textcount):

	abstractList = [['', '']]
	abstract.dropna()
	#print('Abstract :', abstract)

	for sha, abst in zip(sha, abstract):
		abstractList.append([sha, abst]) #allocate each abstract into list

	cleanabstracts = [word for word in abstractList if str(word[1]) != 'nan']
	print('nan removed data :', len(cleanabstracts))
	#print(cleanabstracts[1])
	nlpWork(cleanabstracts, textcount)

def nlpWork(abstract, textcount):
	#print(abstract[0])
	print(len(abstract))

	wordcount = 0
	for words in abstract:
		wordcount +=1
		#print(words[0])
		#print(words[1])
		coronaAnalysis(words[0], nlp(words[1]), wordcount, textcount)
		#coronaAnalysis(sha, nlp(words), wordcount, textcount)
		#bows(words[0], nlp(words[1]), wordcount, textcount)
	print("word count :", wordcount)

	#for ind in len(abstract):
		#print(abstract[i])


def coronaAnalysis(sha, abstract, count, textcount):
	#doc = nlp(text)
	textcount = 0


	cleantext = [t.text for t in abstract if  not t.is_stop  and t.ent_type_ != 'GPE' ] # remove stop words. Exclude Geographic location

	# convert list to nlp doc
	cleandoc = Doc(nlp.vocab, words=cleantext)

	matcher = Matcher(nlp.vocab)

	#print("Search for ", pattern22)
	#matcher.add("medicalcare", None, pattern2, pattern3, pattern4, pattern5)
	#matcher.add("medicalcare", None, pattern2)
	#matcher.add("medicalcare", None, pattern5)
	#matcher.add("medicalcare", None, pattern6)
	matcher.add("medicalcare", None, pattern21)
	matches = matcher(cleandoc)

	for match_id, start, end in matches:

		moveleft = 0
		moveright = 0

		leftwords = []
		rightwords = []

		string_id = nlp.vocab.strings[match_id]  # Get string representation
		span = cleandoc[start:end]  # The matched span
		#print("Span :", span, '\n')
		print(start, end, span.text)
		#print("Len clean Doc :", len(cleandoc))
		#print("Moveleft ", moveleft)
		#print(" Doc Lenght ", len(cleandoc))
		#print(cleandoc[start-1])
		while ((len(cleandoc) >  start + moveleft) and (str(cleandoc[start - moveleft]) != ".") ):

			#print("Prev Word :", cleandoc[start - moveleft])
			leftwords.append(cleandoc[start - moveleft])
			moveleft= moveleft +1
			#print("movement :", moveleftprint("Sum :", end + moveright))

			#print("Sum Left :", start + moveleft)
			if len(cleandoc) ==  start + moveleft:
				break
		leftwords.reverse()
		#print("Left Words :", leftwords)
		#print("Moveright ", moveright)
		#print(" Doc Lenght ", len(cleandoc))

		while ((len(cleandoc) >  end + moveright) and (str(cleandoc[end + moveright]) != ".") ):

			#print("Next Word :", cleandoc[end + moveright])
			moveright = moveright + 1
			#print("movement :", moveright)
			#print("MOVE RIGHT count :", moveright)
			#print("End", end)
			#print("Abstract Length : ", len(abstract))
			#print("Clean Doc Size :", len(cleandoc))

			#print("Sum :", end + moveright)

			if len(cleandoc) ==  end + moveright:
				break
			rightwords.append(cleandoc[end + moveright])

		#rightwords.reverse()
		#print("Right Words :", rightwords)
		combinedList = leftwords + rightwords
		sentence = ' '.join(map(str, combinedList))
		sentence.replace(".","")
		#print("Combined Words ", combinedList, 'SHA ', sha, 'Keyword ', span.text)
		print("Sentence ", sentence, 'SHA ', sha, 'Keyword ', span.text)

		medical_care.append([sha, span.text, sentence])


		#print(start, end, span.text, span.label)
		#print(doc)
		#print(cleandoc)
		#text_list.append([sha, cleandoc])
		#word_dict[span.text] = {}  # create dictionary for keyword
		#word_dict[span.text][cleandoc[start - 1]] = -1
		textcount = +1
	#print(textcount)
	#print(word_list)
	#print(word_dict)
	#print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != '-' and not t.is_stop])
	#print("Document at " , count, cleantext)


def bows(sha, abstract, count, textcount):

	#print("Bag of Words Called")

	#cleanabstract = [t.text for t in abstract if not t.is_stop and t.ent_type_ != 'GPE']  # remove stop words. Exclude Geographic location
	cleanabstract = [t.text for t in abstract if t.ent_type_ != 'GPE']  # remove stop words. Exclude Geographic location
	#print('Abstract :', cleanabstract, '\n')
	#print(len(cleanabstract))
	for word in range(len(cleanabstract)):
		#print(cleanabstract[word])
		allWords.append(cleanabstract[word])
	#allWords.append(cleanabstract)
	#print('Abstract :', allWords, '\n')
	#print(array(allWords).shape)


if __name__ == '__main__':
	print("Process Started!!!")
	start = time.time()

	readfile()
	print(word_dict)
	df = pd.DataFrame(text_list, columns=['sha', 'abstract'])
	#print('Abstract :', text_list, '\n')
	#print('Abstract :', allWords, '\n')

	# create cbow vectors
	#bow_vector = CountVectorizer(tokenizer=allWords, ngram_range=(1, 1))
	#print("Bow Vector :", bow_vector)
	#word_freq = Counter(allWords)
	#most_common = word_freq.most_common(1000000)
	#print("Bag of Word :", most_common)

	#bow = pd.DataFrame(most_common)
	#show top 1000 words
	#print(bow.head(1000))
	#sb.distplot(bow[1])
	#plt.show()
	print("word freq is being written into csv")
	#bow.to_csv('wordfreq.csv', sep=',', index=False)
	df = pd.DataFrame(medical_care, columns=['sha', 'keyword', 'medical_care'])

	df.to_csv('medical_care.csv', sep=',', index=False)
	print("Medicare Data has been written into csv")
	#df.to_csv('vitamin.csv', sep=',', index=False)
	print("Process Ended!!!")
	end = time.time()
	print("Time taken to run the code ", (end - start) // 60, " minutes")
