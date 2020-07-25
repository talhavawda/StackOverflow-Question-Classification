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

dataFrame = pandas.read_csv("SODataset.csv")


"""2. Prepare the Data"""

"""
	Both the csv and pandas library do not recognise the elements in the 'tags' column in the corpus CSV file
		as a list/array of elements. It sees it as a string. 
	So we need to process the data to extract the tags for a question and store them in a list (for that row) and then 
		replace the initial string with this list as the tags for that question
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
			break

		tag = tagsMod[0:commaIndex]
		tagsList.append(tag)
		tagsMod = tagsMod[commaIndex+1:] #'delete' the tag along with its comma | the next comma will now be the first occurrence

	return tagsList

#testing the regex
t = "['iphone', 'objective-c', 'ios', 'cocoa-touch']"
print(tagsAsList(t))




#Display x rows in the dataFrame
	#Using slicing - > [startIndex, endIndex+1] -> the second arg value is exclusive
print(dataFrame[1:5])
print()

#Display x rows in the dataFrame using iloc
	#iloc also uses slicing (but doesnt have to, can put a single index to display data for 1 row)
print(dataFrame.iloc[1:3])
print()

#Display the column heading of the dataFrame
for i in dataFrame:
	print(i)
print()


#get(key) -> Get item(s) from object for given key (e.g. a key can be a  DataFrame column)
#Display rows 0-2 for the 'tags' column
print(dataFrame[0:3].get('tags'))




print()
print("Iteration:")

for i, j in dataFrame[1:5].iterrows():
	print(i, j)

	print(j['tags'])
	tagsList = tagsAsList(j['tags'])
	for i in tagsList:
		print(i, end='\t')
	print("\n============================================================================")
print()

print()

#Display all the rows and columns in the dataFrame (i.e. the entire 2d array/table) [also displays the indexes]
print(dataFrame)
print()

#Display all the rows for the 'tags' column [also displays the indexes]
print(dataFrame['tags'])