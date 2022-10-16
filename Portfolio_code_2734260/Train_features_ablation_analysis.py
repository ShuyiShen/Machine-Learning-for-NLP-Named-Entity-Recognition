import sklearn
import csv
import gensim
import numpy as np
import pandas as pd
import string 
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from nltk.corpus import stopwords

gold = 'data/conll2003.dev-preprocessed.conll'
gold_train = 'data/conll2003.train-preprocessed.conll'
    
def create_vectorizer_and_classifier(features, labels, modelname ='logreg'):
    '''
    Function that takes feature-value pairs and gold labels as input and trains a logistic regression classifier
    
    :param features: feature-value pairs
    :param labels: gold labels
    :type features: a list of dictionaries
    :type labels: a list of strings
    
    :return lr_classifier: a trained LogisticRegression classifier
    :return vec: a DictVectorizer to which the feature values are fitted. 
    '''
    
    if modelname ==  'logreg':
        vec = DictVectorizer()
    #fit creates a mapping between observed feature values and dimensions in a one-hot vector, transform represents the current values as a vector 
        tokens_vectorized = vec.fit_transform(features)
        classifier = LogisticRegression(solver='saga')
        classifier.fit(tokens_vectorized, labels)
        
    elif modelname ==  'NB':
        # TIP: you may need to solve this: https://stackoverflow.com/questions/61814494/what-is-this-warning-convergencewarning-lbfgs-failed-to-converge-status-1
        classifier = MultinomialNB()
        vec = DictVectorizer()
        tokens_vectorized = vec.fit_transform(features)
        classifier.fit(tokens_vectorized, labels)
    
    elif modelname ==  'SVM':
        # TIP: you may need to solve this: https://stackoverflow.com/questions/61814494/what-is-this-warning-convergencewarning-lbfgs-failed-to-converge-status-1
        classifier = svm.LinearSVC(max_iter=2000)
        vec = DictVectorizer()
        tokens_vectorized = vec.fit_transform(features)
        classifier.fit(tokens_vectorized, labels)
        

    
    return classifier, vec
def print_confusion_matrix(predictions, goldlabels):
    '''
    Function that prints out a confusion matrix
    
    :param predictions: predicted labels
    :param goldlabels: gold standard labels
    :type predictions, goldlabels: list of strings
    '''
    
    #based on example from https://datatofish.com/confusion-matrix-python/ 
    data = {'Gold':    goldlabels, 'Predicted': predictions    }
    df = pd.DataFrame(data, columns=['Gold','Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print (confusion_matrix)

  

def print_precision_recall_fscore(predictions, goldlabels):
    '''
    Function that prints out precision, recall and f-score
    
    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions, goldlabels: list of strings
    '''
    
    precision = metrics.precision_score(y_true=goldlabels,
                        y_pred=predictions,
                        average='macro')

    recall = metrics.recall_score(y_true=goldlabels,
                     y_pred=predictions,
                     average='macro')


    fscore = metrics.f1_score(y_true=goldlabels,
                 y_pred=predictions,
                 average='macro')

    print('P:', precision, 'R:', recall, 'F1:', fscore)
    
def crf_annotation(inputfile,annotationcolumn):
    conll_input = pd.read_csv(inputfile, sep='\t', quoting=csv.QUOTE_NONE)
    annotations = conll_input[annotationcolumn].tolist()
    return annotations

inputfile1 = 'data/conll2003.dev.conll_revised'
inputfile2 = 'data/conll2003.dev.crfconll_revised'
goldlabels = crf_annotation(inputfile1, 'gold')
predictions = crf_annotation(inputfile2, 'predict')

print_confusion_matrix(predictions, goldlabels)
print_precision_recall_fscore(predictions, goldlabels)

feature_to_index = {'Token': 0, 'Pos': 1,'Chunklabel': 2,'Prevtoken': 4,'nexttoken': 5,'Prevpos': 6, 'nextpos': 7, 'capital': 8}


def extract_features_and_gold_labels(conllfile, selected_features):
    '''Function that extracts features and gold label from preprocessed conll (here: tokens only).
    
    :param conllfile: path to the (preprocessed) conll file
    :type conllfile: string
    
    
    :return features: a list of dictionaries, with key-value pair providing the value for the feature `token' for individual instances
    :return labels: a list of gold labels of individual instances
    '''
    
    features = []
    labels = []
    conllinput = open(conllfile, 'r')
    #delimiter indicates we are working with a tab separated value (default is comma)
    #quotechar has as default value '"', which is used to indicate the borders of a cell containing longer pieces of text
    #in this file, we have only one token as text, but this token can be '"', which then messes up the format. We set quotechar to a character that does not occur in our file
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    count =0
    for row in csvreader:
        if count == 0:
            pass
            count +=1
        else:    
        #I preprocessed the file so that all rows with instances should contain 10 values, the others are empty lines indicating the beginning of a sentence
            if len(row) == 10:
                #structuring feature value pairs as key-value pairs in a dictionary
                #the first column in the conll file represents tokens
                feature_value = {}
                for feature_name in selected_features:
                    row_index = feature_to_index.get(feature_name)
                    feature_value[feature_name] = row[row_index]
                features.append(feature_value)
                #The last column provides the gold label (= the correct answer). 
                labels.append(row[-1])
                count +=1
    return features, labels

def get_predicted_and_gold_labels(testfile, vectorizer, classifier, selected_features):
    '''
    Function that extracts features and runs classifier on a test file returning predicted and gold labels
    
    :param testfile: path to the (preprocessed) test file
    :param vectorizer: vectorizer in which the mapping between feature values and dimensions is stored
    :param classifier: the trained classifier
    :type testfile: string
    :type vectorizer: DictVectorizer
    :type classifier: LogisticRegression()
    
    
    
    :return predictions: list of output labels provided by the classifier on the test file
    :return goldlabels: list of gold labels as included in the test file
    '''
    
    #we use the same function as above (guarantees features have the same name and form)
    features, goldlabels = extract_features_and_gold_labels(testfile, selected_features)
    #we need to use the same fitting as before, so now we only transform the current features according to this mapping (using only transform)
    test_features_vectorized = vectorizer.transform(features)
    predictions = classifier.predict(test_features_vectorized)
    
    return predictions, goldlabels

#define which from the available features will be used (names must match key names of dictionary feature_to_index)
all_features = ['Token','Prevtoken','capital','Pos','Chunklabel', 'Prevpos','nexttoken','nextpos']

sparse_feature_reps, labels = extract_features_and_gold_labels(gold_train, all_features)
#we can use the same function as before for creating the classifier and vectorizer
lr_classifier, vectorizer = create_vectorizer_and_classifier(sparse_feature_reps, labels, modelname ='logreg')
#when applying our model to new data, we need to use the same features
predictions, goldlabels = get_predicted_and_gold_labels(gold, vectorizer, lr_classifier, all_features)

def classify_data(goldlabels, outputfile, predictions):
    
    """
    :param model: the model returned from three classifiers 
    :param vec: the feature representations transformed by gold features and annotations. 
    :param inputdata: a list of predicted features from system file 
    :param outputfile: the path to our outputfile
    
    The function utilized the pre-defined classifier to predict the annotations from the system file
    (inputdata), and for each pair of testing and predicted features and annotation, the function writes
    them into csv file.
    """
    with open (outputfile, 'w', newline='', encoding = 'utf-8') as outfile:
        count = 0
        for label in goldlabels:
            
            if count == 0:
                
                outfile.write('gold'+'\t'+'predict'+'\n')
                outfile.write(label.rstrip('\n') + '\t'  + predictions[count] + '\n')    
                    
            else:
                
                    
                outfile.write(label.rstrip('\n') + '\t'  + predictions[count] + '\n')  
                    
            count +=1
            
gold = 'data/conll2003.dev-preprocessed.conll'
gold_train = 'data/conll2003.train-preprocessed.conll'

selected_features = ['Token','Prevtoken','nexttoken','Pos', 'Prevpos','nextpos', 'Chunklabel', 'capital']

feature_values, labels = extract_features_and_gold_labels(gold_train, selected_features)

lr_classifier, vectorizer = create_vectorizer_and_classifier(feature_values, labels, modelname ='logreg')

predictions_log, goldlabels_log = get_predicted_and_gold_labels(gold, vectorizer, lr_classifier, selected_features)
# you can uncomment the lines below to test the result 
# print_confusion_matrix(predictions_log, goldlabels_log)
# print_precision_recall_fscore(predictions_log, goldlabels_log)

gold = 'data/conll2003.dev-preprocessed.conll'
gold_train = 'data/conll2003.train-preprocessed.conll'

selected_features = ['Token','Prevtoken','nexttoken','Pos', 'Prevpos','nextpos', 'Chunklabel', 'capital']

feature_values, labels = extract_features_and_gold_labels(gold_train, selected_features)

lr_classifier, vectorizer = create_vectorizer_and_classifier(feature_values, labels, modelname ='logreg')

predictions_log, goldlabels_log = get_predicted_and_gold_labels(gold, vectorizer, lr_classifier, selected_features)
# you can uncomment the lines below to test the result 
# print_confusion_matrix(predictions_log, goldlabels_log)
# print_precision_recall_fscore(predictions_log, goldlabels_log)

svm_classifier, vectorizer = create_vectorizer_and_classifier(feature_values, labels, modelname ='SVM')
predictions_svm, goldlabels_svm = get_predicted_and_gold_labels(gold, vectorizer, svm_classifier, selected_features)
# you can uncomment the lines below to test the result 
# print_confusion_matrix(predictions_svm, goldlabels_svm)
# print_precision_recall_fscore(predictions_svm, goldlabels_svm)

NB_classifier, vectorizer = create_vectorizer_and_classifier(feature_values, labels, modelname ='NB')
predictions_nb, goldlabels_nb = get_predicted_and_gold_labels(gold, vectorizer, NB_classifier, selected_features)
# you can uncomment the lines below to test the result 
# print_confusion_matrix(predictions_nb, goldlabels_nb)
# print_precision_recall_fscore(predictions_nb, goldlabels_nb)

classify_data(goldlabels_log, 'data/conll2003.dev-preprocessed.predictlogistic', predictions_log)
classify_data(goldlabels_svm, 'data/conll2003.dev-preprocessed.predictsvm', predictions_svm)
classify_data(goldlabels_nb,'data/conll2003.dev-preprocessed.predictnb', predictions_nb)

word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)  

gold = 'data/conll2003.dev-preprocessed.conll'
gold_train = 'data/conll2003.train-preprocessed.conll'

def extract_embeddings_as_features_and_gold(conllfile,word_embedding_model):
    '''
    Function that extracts features and gold labels using word embeddings
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    labels = []
    features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    count = 0
    for row in csvreader:
        if count ==0:
            pass
            count+=1
        else:
            if len(row) == 10:
                if row[0] in word_embedding_model:
                    vector = word_embedding_model[row[0]]
                else:
                    vector = [0]*300
                features.append(vector)
                labels.append(row[-1])
    return features, labels

def create_classifier(features, labels, model= 'logreg'):
    '''
    Function that creates classifier from features represented as vectors and gold labels
    
    :param features: list of vector representations of tokens
    :param labels: list of gold labels
    :type features: list of vectors
    :type labels: list of strings
    
    :returns trained logistic regression classifier
    '''
    
    
    if model == 'logreg':
        classifier = LogisticRegression(solver='saga')
        classifier.fit(features, labels)
        
    elif model == 'SVM':
        classifier = svm.LinearSVC(max_iter=2000)
        classifier.fit(features, labels)
    
    return classifier
    
    
def label_data_using_word_embeddings(testfile, word_embedding_model, classifier):
    '''
    Function that extracts word embeddings as features and gold labels from test data and runs a classifier
    
    :param testfile: path to test file
    :param word_embedding_model: distributional semantic model
    :param classifier: trained classifier
    :type testfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    :type classifier: LogisticRegression
    
    :return predictions: list of predicted labels
    :return labels: list of gold labels
    '''
    
    dense_feature_representations, labels = extract_embeddings_as_features_and_gold(testfile,word_embedding_model)
    predictions = classifier.predict(dense_feature_representations)
    
    return predictions, labels


# I printing announcements of where the code is at (since some of these steps take a while)

print('Extracting dense features...')
dense_feature_representations, labels = extract_embeddings_as_features_and_gold(gold_train,word_embedding_model)
print('Training classifier....')
classifier = create_classifier(dense_feature_representations, labels, model= 'SVM')
print('Running evaluation...')
predicted_token, gold_token = label_data_using_word_embeddings(gold, word_embedding_model, classifier)

# you can uncomment the lines below to test the result 
# print_confusion_matrix(predicted_token, gold_token)
# print_precision_recall_fscore(predicted_token, gold_token)

def extract_embeddings_of_current_and_preceding_as_features_and_gold(conllfile,word_embedding_model):
    '''
    Function that extracts features and gold labels using word embeddings for current and preceding token
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    labels = []
    features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    count = 0
    for row in csvreader:
        if count ==0:
            pass
            count+=1
        else:    
            if len(row) == 10:
                if row[0] in word_embedding_model:
                    vector1 = word_embedding_model[row[0]]
                else:
                    vector1 = [0]*300
                if row[4] in word_embedding_model:
                    vector2 = word_embedding_model[row[4]]
                else:
                    vector2 = [0]*300
                features.append(np.concatenate((vector1,vector2)))
                labels.append(row[-1])
    return features, labels
    
    
def label_data_using_word_embeddings_current_and_preceding(testfile, word_embedding_model, classifier):
    '''
    Function that extracts word embeddings as features (of current and preceding token) and gold labels from test data and runs a trained classifier
    
    :param testfile: path to test file
    :param word_embedding_model: distributional semantic model
    :param classifier: trained classifier
    :type testfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    :type classifier: LogisticRegression
    
    :return predictions: list of predicted labels
    :return labels: list of gold labels
    '''
    
    features, labels = extract_embeddings_of_current_and_preceding_as_features_and_gold(testfile,word_embedding_model)
    predictions = classifier.predict(features)
    
    return predictions, labels


print('Extracting dense features...')
features, labels = extract_embeddings_of_current_and_preceding_as_features_and_gold(gold_train,word_embedding_model)
print('Training classifier...')

classifier = create_classifier(features, labels, model='SVM')
print('Running evaluation...')
predicted_pretoken, gold_pretoken = label_data_using_word_embeddings_current_and_preceding(gold, word_embedding_model, classifier)

# you can uncomment the lines below to test the result 
# print_confusion_matrix(predicted_pretoken, gold_pretoken)
# print_precision_recall_fscore(predicted_pretoken, gold_pretoken)

def extract_word_embedding(token, word_embedding_model):
    '''
    Function that returns the word embedding for a given token out of a distributional semantic model and a 300-dimension vector of 0s otherwise
    
    :param token: the token
    :param word_embedding_model: the distributional semantic model
    :type token: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :returns a vector representation of the token
    '''
    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0]*300
    return vector


def extract_feature_values(row, selected_features):
    '''
    Function that extracts feature value pairs from row
    
    :param row: row from conll file
    :param selected_features: list of selected features
    :type row: string
    :type selected_features: list of strings
    
    :returns: dictionary of feature value pairs
    '''
    feature_values = {}
    for feature_name in selected_features:
        r_index = feature_to_index.get(feature_name)
        feature_values[feature_name] = row[r_index]
        
    return feature_values
    
    
def create_vectorizer_traditional_features(feature_values):
    '''
    Function that creates vectorizer for set of feature values
    
    :param feature_values: list of dictionaries containing feature-value pairs
    :type feature_values: list of dictionairies (key and values are strings)
    
    :returns: vectorizer with feature values fitted
    '''
    vectorizer = DictVectorizer()
    vectorizer.fit(feature_values)
    
    return vectorizer
        
    
def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''
    Function that takes sparse and dense feature representations and appends their vector representation
    
    :param dense_vectors: list of dense vector representations
    :param sparse_features: list of sparse vector representations
    :type dense_vector: list of arrays
    :type sparse_features: list of lists
    
    :returns: list of arrays in which sparse and dense vectors are concatenated
    '''
    
    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())
    
    
    for index, vector in enumerate(sparse_vectors):
        
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)
        
    return combined_vectors
    

def extract_traditional_features_and_embeddings_plus_gold_labels(conllfile, word_embedding_model, vectorizer=None):
    '''
    Function that extracts traditional features as well as embeddings and gold labels using word embeddings for current and preceding token
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    labels = []
    dense_vectors = []
    traditional_features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    count = 0
    for row in csvreader:
        if count ==0:
            pass
            count +=1
        else:
            if len(row) == 10:
                token_vector = extract_word_embedding(row[0], word_embedding_model)
                pt_vector = extract_word_embedding(row[4], word_embedding_model)
                dense_vectors.append(np.concatenate((token_vector,pt_vector)))
                #mixing very sparse representations (for one-hot tokens) and dense representations is a bad idea
                #we thus only use other features with limited values
                other_features = extract_feature_values(row, ['capital','Pos','Chunklabel'])
                traditional_features.append(other_features)
                #adding gold label to labels
                labels.append(row[-1])
            
    #create vector representation of traditional features
    if vectorizer is None:
        #creates vectorizer that provides mapping (only if not created earlier)
        vectorizer = create_vectorizer_traditional_features(traditional_features)
    sparse_features = vectorizer.transform(traditional_features)
    combined_vectors = combine_sparse_and_dense_features(dense_vectors, sparse_features)
    
    return combined_vectors, vectorizer, labels

def label_data_with_combined_features(testfile, classifier, vectorizer, word_embedding_model):
    '''
    Function that labels data with model using both sparse and dense features
    '''
    feature_vectors, vectorizer, goldlabels = extract_traditional_features_and_embeddings_plus_gold_labels(testfile, word_embedding_model, vectorizer)
    predictions = classifier.predict(feature_vectors)
    
    return predictions, goldlabels


print('Extracting Features...')
feature_vectors, vectorizer, gold_labels = extract_traditional_features_and_embeddings_plus_gold_labels(gold_train, word_embedding_model)
print('Training classifier....')
lr_classifier = create_classifier(feature_vectors, gold_labels, model='logreg')
print('Running the evaluation...')
predictions_mix, goldlabels_mix = label_data_with_combined_features(gold, lr_classifier, vectorizer, word_embedding_model)
# print_confusion_matrix(predictions_mix, goldlabels_mix)
# print_precision_recall_fscore(predictions_mix, goldlabels_mix)

print('Extracting Features...')
feature_vectors, vectorizer, gold_labels = extract_traditional_features_and_embeddings_plus_gold_labels(gold_train, word_embedding_model)
print('Training classifier....')
svm_classifier = create_classifier(feature_vectors, gold_labels, model='SVM')
print('Running the evaluation...')
predictions_mix_svm, goldlabels_mix_svm = label_data_with_combined_features(gold, svm_classifier, vectorizer, word_embedding_model)
#print_confusion_matrix(predictions_mix_svm, goldlabels_mix_svm)
#print_precision_recall_fscore(predictions_mix_svm, goldlabels_mix_svm)

classify_data(gold_token, 'data/conll2003.dev-preprocessed.predict_w2vtoken', predicted_token)
classify_data(gold_pretoken, 'data/conll2003.dev-preprocessed.predict_w2vpretoken', predicted_pretoken)
classify_data(goldlabels_mix, 'data/conll2003.dev-preprocessed.predict_w2vmix', predictions_mix)