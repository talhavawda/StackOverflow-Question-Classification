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
	So we need to process the data to extract each tag (for a row) and store them in a list (for that row) and then 
		store this list as the tags for that question (replacing the initial string)
"""


def tagsAsList(tags: str):
	"""
	:returns: the tags for a StackOverflow question as a list
	"""
	tagsList = [] #initialise the list to store the tags for a StackOverflow question (a row in the corpus) | the question is under the 'title' tag

	"""
		Use a regular expression to remove the opening and closing square brackets, and the apostrophes
		The commas separating the tags remains, so as to identify the different tags
	"""
	RE = r'[\[\]\']'  # Regular Expression to match non-alphanumeric and non-whitespace characters
	tags = re.sub(RE, "", tags)
	return  tags

#testing the regex
t = "['iphone', 'objective-c', 'ios', 'cocoa-touch']"
print(tagsAsList(t))





print(dataFrame[1:5]) #startIndex, endIndex (exclsuive)
print(dataFrame.iloc[1:3])
tag = []
t = dataFrame[1:3].get(2)

tag.append(t)

print()
print(t)
print()
print("Iteration:")

for i, j in dataFrame[1:5].iterrows():
	print(i, j)

	print(j['tags'])
print()
print(dataFrame)
