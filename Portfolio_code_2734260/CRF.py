import sklearn
import csv
import sys
import sklearn_crfsuite
from sklearn_crfsuite import metrics


def token2features(sentence, i):

    token = sentence[i][0]
    postag = sentence[i][1]
    
    features = {
        'bias': 1.0,
        'token': token.lower(),
        'postag': postag
    }
    if i == 0:
        features['BOS'] = True
    elif i == len(sentence) -1:
        features['EOS'] = True
    
    
    return features

def sent2features(sent):
    return [token2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    #if you added features to your input file, make sure to add them here as well.
    #print([ner for token, postag, chunklabel, label, pretoken, nexttoken, prepos,nextpos, capital, ner in sent])
    return [label for token, postag, chunklabel, label in sent]

def sent2tokens(sent):
    return [token for token, postag, chunklabel, label  in sent]
    
    
def extract_sents_from_conll(inputfile):
    sents = []
    current_sent = []

    with open(inputfile, 'r') as my_conll:
        for line in my_conll:
            row = line.strip("\n").split('\t')
            
            if len(row) == 1:
                 sents.append(current_sent)
                 current_sent = []
            else:
                current_sent.append(tuple(row))
    
    return sents


def train_crf_model(X_train, y_train):

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    
    return crf

def create_crf_model(trainingfile):

    train_sents = extract_sents_from_conll(trainingfile)
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    crf = train_crf_model(X_train, y_train)
    
    return crf


def run_crf_model(crf, evaluationfile):

    test_sents = extract_sents_from_conll(evaluationfile)
    X_test = [sent2features(s) for s in test_sents]
    y_pred = crf.predict(X_test)
    
    return y_pred, X_test

def write_out_evaluation(eval_data, pred_labels, outputfile):

    outfile = open(outputfile, 'w')
    
    for evalsents, predsents in zip(eval_data, pred_labels):
        for data, pred in zip(evalsents, predsents):
            outfile.write(data.get('token') + "\t" + pred + "\n")

def train_and_run_crf_model(trainingfile, evaluationfile, outputfile):

    crf = create_crf_model(trainingfile)
    pred_labels, eval_data = run_crf_model(crf, evaluationfile)
    write_out_evaluation(eval_data, pred_labels, outputfile)

def main(argv):
    
    #argv = sys.argv
    
    
    trainingfile = argv[1]
    evaluationfile = argv[2]
    outputfile = argv[3]
    
    train_and_run_crf_model(trainingfile, evaluationfile, outputfile)

    
gold = 'data/conll2003.dev.conll'
gold_train = 'data/conll2003.train.conll'
argv = ['mypython_program',gold_train, gold, 'data/conll2003.dev.crfconll']

if __name__ == '__main__':
    main(argv)
    
    
inputdata = 'data/conll2003.dev.crfconll'
outputfile = 'data/conll2003.dev.crfconll_revised'

inputdata1 = 'data/conll2003.dev.conll'
outputfile1 = 'data/conll2003.dev.conll_revised'

def narrow_file(inputdata,outputfile, mode ='crf'):
    """
    :para inputdata: the path to predicted output (conll2003 dev.crfonll) file 
     and original test file (conll2003.dev)
    :para outputfile: the path to the outputfile 
    
    This function will return an outputfile with added header, so that we can use 
    pandas to read the file easily. 
    
    """
    count = 0
    with open (outputfile, 'w', newline='', encoding = 'utf-8') as outfile:
        with open(inputdata, 'r', encoding='utf8') as infile:
            for line in infile:
                if count ==0:
                    if mode == 'crf':
                        outfile.write('token'+'\t'+'predict' +'\n') 
                        outfile.write(line.rstrip('\n') +'\n')  
                    else:
                        outfile.write('token'+'\t'+'pos'+'\t'+'chunklabel'+'\t'+'gold' +'\n') 
                        outfile.write(line.rstrip('\n') +'\n')  
                else:
                    if len(line.rstrip('\n').split())>0:
                        outfile.write(line.rstrip('\n') +'\n') 
                    else:
                        pass
                count +=1
                

narrow_file(inputdata ,outputfile, mode='crf')    
narrow_file(inputdata1 ,outputfile1, mode='dev')  