import pandas as pd
import os
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")

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

word_list = []
word_dict = {}

text_list = [['', '']]

customize_stop_words = [
    'From','from', 'To', 'to', 'Hospital', 'hospital', '.', '-', ')', '(', ',', ':', 'of', 'for', 'the', 'The', 'is',
	'[', ']'
]
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True

os.chdir("/home/saul/corona/CORD-19-research-challenge/2020-03-13")

filename = 'all_sources_metadata_2020-03-13.csv'


def readfile():

	textcount = 0
	print("reading the file")
	coronafile = pd.read_csv(filename, sep=',')
	#print(f'number of rows {len(coronafile)}')
	#print(f'columns {coronafile.columns}')
	#print(coronafile.head(5))

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
		#coronaAnalysis(sha, nlp(words), wordcount, textcount)
		bows(words[0], nlp(words[1]), wordcount, abstract, textcount)
	print("word count :", wordcount)

	#for ind in len(abstract):
		#print(abstract[i])


def coronaAnalysis(sha, doc, count, textcount):
	#doc = nlp(text)
	textcount = 0


	cleantext = [t.text for t in doc if  not t.is_stop  and t.ent_type_ != 'GPE' ] # remove stop words. Exclude Geograpic location

	# convert list to nlp doc
	cleandoc = Doc(nlp.vocab, words=cleantext)

	matcher = Matcher(nlp.vocab)

	#matcher.add("medicalcare", None, pattern2, pattern3, pattern4, pattern5)
	#matcher.add("medicalcare", None, pattern2)
	#matcher.add("medicalcare", None, pattern5)
	#matcher.add("medicalcare", None, pattern6)
	matcher.add("medicalcare", None, pattern10)
	matches = matcher(cleandoc)


	#print(matches)
	for match_id, start, end in matches:
		string_id = nlp.vocab.strings[match_id]  # Get string representation
		span = cleandoc[start:end]  # The matched span
		#print(start, end, span.text, span.label)
		#print(doc)
		#print(cleandoc)
		text_list.append([sha, cleandoc])
		#word_dict[span.text] = {}  # create dictionary for keyword
		#word_dict[span.text][cleandoc[start - 1]] = -1
		textcount = +1
	#print(textcount)
	#print(word_list)
	#print(word_dict)
	#print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != '-' and not t.is_stop])
	#print("Document at " , count, cleantext)


def bows(sha, abstract,  doc, count, textcount):
	print("Bag of Words Called")
	print('Abstract :', abstract, '\n')





if __name__ == '__main__':
	readfile()
	print(word_dict)
	df = pd.DataFrame(text_list, columns=['sha', 'abstract'])

	df.to_csv('vitamin.csv', sep=',', index=False)