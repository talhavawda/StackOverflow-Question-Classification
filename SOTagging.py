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
			break

		tag = tagsMod[0:commaIndex]
		tagsList.append(tag)
		tagsMod = tagsMod[commaIndex+1:] #'delete' the tag along with its comma | the next comma will now be the first occurrence

	return tagsList


"""
	Extract individual tags and store them as a list of tags for that row's 'tags' value
		by using the tagsAsList() function defined above
	
	We can use the pandas function apply() to do this for us on (a column of) the entire dataFrame at once
		instead of us iterating through each row, and applying the tagsAsList() to the 'tags' value for that row
	apply() takes in the function that we want to apply to all the values of a column as an ARGUMENT -> we are not 
		calling the function thus not putting the round brackets after the function name
	
"""
dataFrame['tags'] = dataFrame['tags'].apply(tagsAsList)


"""
	Now, even though when displaying the dataFrame the values in the 'tags' column look the 
		exact same as before (opening and closing square brackets enclosing the list, each tag within single quotes, 
		and tags separated with commas), they are instead STORED as list instead of a string. 
	Those extra symbols are added when printing a list(these were initially part of the actual string as characters,
		which is what we dont't want) 
"""



"""
	All the tags (in the tags column) in the coprus are in lowercase but the questions (in the title column) are in a 
		mixture of both uppercase and lowercase
	So we need to convert all the text in the 'title' column to lowercase
"""
dataFrame['title'] = dataFrame['title'].str.lower()

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

for i, j in dataFrame.iterrows(): #i is the index; j is the columns(panda Series')
	print(j['tags'])
	#tagsList = tagsAsList(j['tags'])
	#j['tags'] = tagsList
	for i in j['tags']:
		print(i, end='\t')
	print("\n============================================================================")
print()

print()

#Display all the rows and columns in the dataFrame (i.e. the entire 2d array/table) [also displays the indexes]
print(dataFrame)
print()

#Display all the rows for the 'tags' column [also displays the indexes]
print(dataFrame['tags'])

