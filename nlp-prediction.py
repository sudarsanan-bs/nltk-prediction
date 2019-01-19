import os
import nltk

#***************************************
# Author: Sudarsanan B S
#***************************************

# Classifier engine to predict if a given name is that of a male or that of a female
# Feature set used is the last letter of name

# Feature extraction method, returns last letter of name
def gender_features(word):
    return {'last_letter': word[-1]}

# Load data and training 

path = os.getcwd()
male = os.path.join(path ,'male.txt')
female = os.path.join(path ,'female.txt')

male_lines = [line.rstrip('\n') for line in open(male)]
female_lines = [line.rstrip('\n') for line in open(female)]

names = ([(name, 'male') for name in male_lines] + 
 	 [(name, 'female') for name in female_lines])

# Defining feature set
featuresets = [(gender_features(n), g) for (n,g) in names]
train_set = featuresets

# Train using Naive Bayes Classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Predict
query_name = input("Name: ")
print(query_name + ' is a ' + classifier.classify(gender_features(query_name)))
