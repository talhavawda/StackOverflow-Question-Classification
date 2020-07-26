"""
	COMP316 PROJECT 2020

	Talha Vawda (218023210)

	Language Classification (Tagging) of Stack Overflow Questions

	Acknowledgements:
		1. GeeksForGeeks tutorials on pandas' DataFrame

"""
import csv
import pandas
import  re #regular expression module
import nltk


"""1. Obtain corpus"""

#Using csv library
"""
with open("SODataset.csv") as corpusFile:
	#c = csv.reader(corpusFile)
	c = csv.DictReader(corpusFile)
	for r in c:
		print(f'{", ".join(r)}')
		print(r)
		print(f"{r['title']}\t{r['tags']}", end='\t\t')
		print()

"""


#Using pandas library
corpusDF = pandas.read_csv("SODataset.csv") #corpusDF is the DataFrame structure representing the corpus
print("Corpus has been read in as a DataFrame structure")


"""2. Prepare the Data"""


"""
	Both the csv and pandas library do not recognise the elements in the 'tags' column in the corpus CSV file
		as a list/array of elements. It sees it as a string. 
	So we need to process the data to extract the individual tags for a question and store them in a list (for that row)
		and then replace the initial string with this list as the tags for that question
"""
def tagsAsList(tags: str):
	"""
	:param tags: the tags for a StackOverflow question as a string
	:returns: the tags for a StackOverflow question as a list of strings
	"""

	tagsList = [] #initialise the list to store the tags for a StackOverflow question (a row in the corpus) | the question is under the 'title' tag

	"""
		Use a regular expression to remove the opening and closing square brackets, the apostrophes, and whitespace
			by substituting them with an empty string 
		The commas separating the tags remains, so as to identify the different tags
	"""
	regex = r'[\[\]\'\s]'
	tagsMod = re.sub(regex, "", tags) #modified tags string with unneccesary symbols removed using the above regex

	noTags = tagsMod.count(',') + 1 #number of tags in tags string

	for i in range(noTags):
		try:
			commaIndex = tagsMod.index(',') #first occurrence of a comma in the tags string
		except ValueError: #ValueError thrown when substring ',' is not found in tagsMod -> this means that there is only one tag left
			tag = tagsMod
			tagsList.append(tag)
			break #We are done with all the tags so exit the loop (don't execute the remaining loop code) to return the tagsList

		tag = tagsMod[0:commaIndex]
		tagsList.append(tag)
		tagsMod = tagsMod[commaIndex+1:] #'delete' the tag along with its comma | the next comma will now be the first occurrence

	return tagsList


"""
	Extract individual tags and store them as a list of tags for that row's 'tags' value
		by using the tagsAsList() function defined above
	
	We can use the pandas function apply() to do this for us on (a column of) the entire corpusDF at once
		instead of us iterating through each row, and applying the tagsAsList() to the 'tags' value for that row
	apply() takes in the function that we want to apply to all the values of a column as an ARGUMENT -> we are not 
		calling the function thus not putting the round brackets after the function name
	
"""
corpusDF['tags'] = corpusDF['tags'].apply(tagsAsList)
print("Tags have been stored as a list for each StackOverflow question")


"""
	Now, even though when displaying the corpusDF, the values in the 'tags' column look the 
		exact same as before (opening and closing square brackets enclosing the list, each tag within single quotes, 
		and tags separated with commas), they are instead STORED as a list instead of a string. 
	Those extra symbols are added when printing a list(these were initially part of the actual string as characters,
		which is what we dont't want) 
"""


"""
	All the tags (in the tags column) in the coprus are in lowercase but the questions (in the title column) are in a 
		mixture of both uppercase and lowercase
	So we need to convert all the text in the 'title' column to lowercase
"""
corpusDF['title'] = corpusDF['title'].str.lower()
print("All question titles have been converted to lowercase")


"""
	Remove Unnecessary Punctuation
		Will not be removing all punctuation as many punctuation symbols are part of programming syntax/jargon
	
	Some question titles don't end in a question mark whilst others do.
	To ensure uniformity, we are going to remove question marks from the question titles
	
	Many of the questions contain a colon - it is mostly used in the question to explain what the question
		is about after mentioning a programming language or term (e.g. a function, error, etc.)
		E.g.:
			"AttributeError: 'NoneType' object has no attribute 'split'"
			"JavaScript: How can I insert a string at a specific index"
	Thus the colon is unnecessary and should be removed
"""
def cleanPunct(question : str):
	"""
		:param question: a StackOverflow question as a string
		:returns: the question with question mark(s) and colon(s) removed (if it had any)
	"""
	# return question.replace('?', '')
	return re.sub(r'[\?\:]', '', question)

#print(corpusDF[23:24].values)
corpusDF['title'] = corpusDF['title'].apply(cleanPunct)
print("Unnecessary punctuation has been removed from question titles")
#print(corpusDF[23:24].values)

"""
	
"""

"""Display information about the corpus"""

tagsList = [] #list of all the tags used (will contain duplicates)

for questionTags in corpusDF['tags'].values: #questionTags is a list of all the tags for that question(row)
	for tag in questionTags:
		tagsList.append(tag)

tagsSet = set(tagsList) #convert tagsList to a set to only store unique tags (one instance of each tag)
uniqueTagsList = list(tagsSet)

print("\nInformation about the corpus:")
print("\tNumber of questions: ", len(corpusDF), sep='\t\t')     #100000
print("\tTotal number of tags used: ", len(tagsList), sep='\t') #194219
print("\tNumber of unique tags: ", len(tagsSet), sep='\t\t')    #100


"""
#Displaying a frequency distribution  of the tags
tagsFD = nltk.FreqDist(tagsList)
from matplotlib import pyplot
pyplot.style.use('bmh')
fig, ax = pyplot.subplots(figsize=(15, 10))
tagsFD.plot(100, cumulative=False)
"""


print()


#Display x rows in the corpusDF
	#Using slicing - > [startIndex, endIndex+1] -> the second arg value is exclusive
print(corpusDF[1:5])
print()

#Access/display the actual values of x rows in the corpusDF (WITHOUT the indexes and column headings)
	#each row is a list/array, with the values for the columns (for that row) being separated by a space
	#each row is also an element of a list of all the rows -> 2D array
		#However, in our case for corpusDF, since the element in the 2nd column for a row is a List of tags, we end up with a 3D Array
			#But if we looping through corpusDF['tags'].values only, then it'll be a 2D array as now the elements of a row can be seen as the tags (since we ignoring the title)
	#since these are list, we can traverse through them using for loops
print(corpusDF[1:5].values)
print()

#Display x rows in the corpusDF using iloc
	#iloc also uses slicing (but doesnt have to, can put a single index to display data for 1 row)
print(corpusDF.iloc[1:3])
print()

#Display the column heading of the corpusDF
for i in corpusDF:
	print(i)
print()


#get(key) -> Get item(s) from object for given key (e.g. a key can be a  DataFrame column)
#Display rows 0-2 for the 'tags' column
print(corpusDF[0:3].get('tags'))




print()
print("Iteration:")

for i, j in corpusDF[0:5].iterrows(): #i is the index; j is the columns(panda Series')
	print(j['tags'])
	#tagsList = tagsAsList(j['tags'])
	#j['tags'] = tagsList
	for i in j['tags']:
		print(i, end='\t')
	print("\n============================================================================")
print()

print()

#Display all the rows and columns in the corpusDF (i.e. the entire 2d array/table) [also displays the indexes]
print(corpusDF)
print()

#Display all the rows for the 'tags' column [also displays the indexes]
print(corpusDF['tags'])

print()
print(corpusDF.info())

print()
print('Duplicate entries: {}'.format(corpusDF['title'].duplicated().sum()))


