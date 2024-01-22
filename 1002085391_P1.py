import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from math import log10, sqrt


def stemmed_tokens(tokens, stopwords): #function to stem the tokens with parameters tokens and stopwords
    stemmed = [] #created an empty stem list
    for token in tokens: #runs a loop to go through each token after using RegexpTokenizer 
        if token not in stopwords: # Condition to remove the stopwords
            stemmed.append(stemmer.stem(token)) # appends the tokenized words after removing the stopwords
    return stemmed #returns the stemmed words after removing the stopwords


def tokenizing(document): #function to tokenise the document with parameter as document
    document = document.lower() #we lower all the value of document
    tokens = tokenizer.tokenize(document) #tokenise the word using  module ntlk.tokenise and class RegexpTokenizer
    list_of_stopping_words = stopwords.words("english") #making the list of stopword using module ntlk.corpus and class stopwords
    tokens_stemmed = stemmed_tokens(tokens, list_of_stopping_words) #calling the stemmed tokens function to removed stopped words and then stemming it
    return tokens_stemmed #returning tokens which are stemmed and free of stopwords


corpusroot = (r"D:\Personal\Academics\UTA\3rd semester\Data Mining\P1\P1\US_Inaugural_Addresses") #initializing the path to the documents 
stemmer = PorterStemmer() #assigning the typr of stemming
tokenizer = RegexpTokenizer(r"[a-zA-Z]+") #assigning the type of tokenizer
document_term_frequency = {} #dictionary to keep term frequency of each word for all the documents
document_frequency = Counter() #Counter dictionary to keep the count of words in through out the documents
normalised_weights = {} #dictionary to keep the normalised weights of the terms
length_of_weight = {} #dictionary to store the of weight for a particular word 
inverted_index = {} #dictoranry with inverse index of normalised weights


def idf(token):#function to get idf without stemming the token and using parameters token
    if document_frequency[token] == 0: #condition to check whether the document frequncy is zero or not
        return -1 #if document frequncy is zero it will return -1
    else:
        Total_number_of_documents = len(document_term_frequency) #to find the total number of documents
        idf_calculated = log10(Total_number_of_documents / document_frequency[token]) #formula to find the IDF
        return idf_calculated #returning the calculated IDF

def getidf(token): #function to get the IDF but by stemming the token and using parameters token
    tok = ''.join(tokenizing(token)) #using to join to remove it from list and calling tokenization function to stem the word
    if document_frequency[tok] == 0: #condition to check whether the document frequncy is zero or not
        return -1 #if document frequncy is zero it will return -1
    else:
        Total_number_of_documents = len(document_term_frequency) #to find the total number of documents
        idf = log10(Total_number_of_documents / document_frequency[tok]) #formula to find the IDF
        return idf #returning the calculated IDF

def gettf(filename, token): #function to get term frequency bu using parameters filename and token
    if document_term_frequency[filename][token] == 0: #check's for condition whether term frequency for that term is zero or not 
        return 0 # returns the value as zero if the term frequency is zero
    else:
        tf = 1 + log10(document_term_frequency[filename][token]) #if its not zero it calculates for term frequency using its formula
        return tf #returns the calculated term frequency


def gettfidf(filename, token): #function to calculate to product of tf and idf with filename and token as its parameters
    tf = gettf(filename, token) #calling the function to get the tf value
    inverse_df = idf(token) #calling the function to get the IDF value
    tfidf = tf * inverse_df #calculating tf-idf value
    return tfidf #returning tf-idf value


def normalise(): #function to normalise the weights
    for filename in document_term_frequency: #starting the loop to go through different files
        normalised_weights[filename] = Counter() #initializing the normalised weights dictionary to counter dict values
        weight_squared = 0 #initialised the sqaured value of weight to 0
        for token in document_term_frequency[filename]: #starting to the loop to go through different tokens in term frequency in that file
            weight = gettfidf(filename, token) #getting the tf-idf weight which isnt normalised
            normalised_weights[filename][token] = weight #putting the above found weight in normalised weights counter dict
            weight_squared += weight**2 #finding the squared value of the weight which will be used to normalise the weight
        length_of_weight[filename] = sqrt(weight_squared) #finding the length of the weight which is the squared root value of the above
        for token in normalised_weights[filename]: #starting a for loop to go through the stored weights
            normalised_weights[filename][token] = (normalised_weights[filename][token] / length_of_weight[filename]) #making the weights normalised by dividing the stored weight by its length
            if token not in inverted_index: #checks for condition if the token is present in the inverted index dictionary
                inverted_index[token] = Counter() #if it is not present then it makes it a counter dictionary
            inverted_index[token][filename] = normalised_weights[filename][token] #makes a replica of normalised weights but with inverted index


def getweight(filename, token): #function to get normalised weight with filename and token as its parameters
    tok = ''.join(tokenizing(token)) #stemming the token and removing it from the list
    normalise() #calling the normalise function   
    if filename in document_term_frequency: #checks if file exists in term frequency dictionary
        if document_term_frequency[filename][tok] == 0: # checks if the token exists in that file
            return 0 #returns 0 if it doesnt exist
        else:
            return normalised_weights[filename][tok] #assigning the weights for that file and token and returning the weights
    else:
        return 0 #returns 0 if filename is not present


def cosine_similarity(documents, query_term_frequency, query_length): #function to find cosine similarity between query and document
    similarity = Counter() #initaliazing similarity to a counter
    for doc in normalised_weights: #starting a lopop to go through all the documents
        cosine_sim = 0 #initialising cosine similarity as zero
        for token in query_term_frequency: #starting a loop to go trough token in query
            query_normalised = query_term_frequency[token] / query_length #calculating normalised weight of the query
            if doc in documents[token]: #checking if the document is present in document dictionary created in query function
                document_normalised = inverted_index[token][doc] #assigning the normalised weight for the document with that token to a variable
                cosine_sim += query_normalised * document_normalised #calcuating and summing the cosine similarity
        similarity[doc] = cosine_sim #storing the calculated cosine similairty to the counter dictionary of similarity
    if all(score==0 for doc, score in similarity.items()): #checking if all the scores are 0 or not in similarity dictionary
        result = 'No Search Found' #if all the scores are zero result is the search was not found
        score = 0 #the score of the failed search is 0
    else:
        highest = similarity.most_common(1) #or else we take the highest score in the similarity dictionary
        result, score = highest[0] #assigned result which is the file name and highest score of that search
    return result, score #returning the result and the score


def query(qstring): #function to search a query string in a folder contianing files
    sentence = qstring.lower() #lowering the query string
    query_term_frequency = {} #creating a term frequency dictionary for query
    query_length = 0 #initialising the length of query to zero
    documents = {} # creating a dictionary to store filname for that token in query
    sentence = tokenizing(sentence) #stemming the query
    normalise() #calling normalisation function
    for token in sentence: #starting the loops to go through differet tokens in the query
        if token not in inverted_index: #if token is not already presented in inverted index weight dictionary
            inverted_index[token]=Counter({'None':0}) #then add that token to the dictionary
            normalised_weights['None']= Counter({token:0}) #add it to the  normalised weights dictionary too
            documents[token], weights = zip(*inverted_index[token].most_common()) #stored the none as filename in documents
        else:
            documents[token], weights = zip(*inverted_index[token].most_common()) #else stored the real file name in documents which has their token in the  inverted index dict
        count = sentence.count(token) #count the number of times the token is there
        query_term_frequency[token] = 1 + log10(count) #calculate the query's token term frequency
        query_length += query_term_frequency[token] ** 2 #finding the length of the above to normalise it
    query_length = sqrt(query_length) #finding the square root of above
    result, score = cosine_similarity(documents, query_term_frequency, query_length) #finding the cosine similarity between the query and the document
    return result, score #returning the result and the score of they query function


for filename in os.listdir(corpusroot): #loop to go through different files in the corpusroot path
    if filename.startswith("0") or filename.startswith("1"): ##checks if the file name starts with 0 or 1 to go forward with the operation
        file = open(os.path.join(corpusroot, filename), "r", encoding="windows-1252") #opening the file while also incorporating the filename in text in read mode
        doc = file.read() #assigning the open file in read mode to a variable doc
        file.close() #closing the file which was opened earlier
        tokens = tokenizing(doc) #tokenizing the document which we read
        document_frequency += Counter(list(set(tokens))) #adding to the already defined document frequency by first making the token unique and then converting to the list and making it a counter
        document_term_frequency_per_file = Counter(tokens) #creating a temp Counter dictionary to keep the count of words in that document
        document_term_frequency[filename] = document_term_frequency_per_file.copy() #coping the temp Counter dict to already created term frequency dictionary with filname as its keys
        document_term_frequency_per_file.clear() #clearing the temporary created counter dictionary

print("%.12f" % getidf("british"))
print("%.12f" % getidf("union"))
print("%.12f" % getidf("war"))
print("%.12f" % getidf("military"))
print("%.12f" % getidf("great"))
print("--------------")
print("%.12f" % getweight("02_washington_1793.txt", "arrive"))
print("%.12f" % getweight("07_madison_1813.txt", "war"))
print("%.12f" % getweight("12_jackson_1833.txt", "union"))
print("%.12f" % getweight("09_monroe_1821.txt", "british"))
print("%.12f" % getweight("05_jefferson_1805.txt", "public"))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))
