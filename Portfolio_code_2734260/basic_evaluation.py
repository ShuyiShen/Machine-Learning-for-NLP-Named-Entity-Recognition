import sys
import pandas as pd
from collections import defaultdict, Counter
from nose.tools import assert_equal
import csv

def extract_annotations(inputfile, annotationcolumn, delimiter='\t'):
    '''
    This function extracts annotations represented in the conll format from a file
    
    :param inputfile: the path to the conll file
    :param annotationcolumn: the name of the column in which the target annotation is provided
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :type inputfile: string
    :type annotationcolumn: string
    :type delimiter: string
    :returns: the annotations as a list
    '''
   
    conll_input = pd.read_csv(inputfile, sep=delimiter,quoting=csv.QUOTE_NONE )
    annotations = conll_input[annotationcolumn].tolist()
    return annotations

def obtain_counts(goldannotations, machineannotations):
    '''
    This function compares the gold annotations to machine output
    
    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: the type of the object created in extract_annotations
    :type machineannotations: the type of the object created in extract_annotations
    
    :returns: a countainer providing the counts for each predicted and gold class pair
    '''
    
  
    evaluation_counts = defaultdict(Counter)
    for gold,sys in zip(goldannotations, machineannotations):
        if gold == sys:
            evaluation_counts[gold][sys] += 1
        else:     
            evaluation_counts[gold][sys] += 1
    return evaluation_counts
    
def calculate_precision_recall_fscore(evaluation_counts):
    '''
    Calculate precision recall and fscore for each class and return them in a dictionary
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :returns the precision, recall and f-score of each class in a container
    '''
    
    evaluation_mini = defaultdict(Counter)
    for keys,values in evaluation_counts.items():
        for key, value in values.items():
                evaluation_mini[keys]['TP'] =0
                evaluation_mini[keys]['FN'] =0
                evaluation_mini[keys]['FP'] =0

    for keys,values in evaluation_counts.items():
        for key, value in values.items():

                if keys == key:
                    evaluation_mini[keys]['TP'] += value
                else:     
                    evaluation_mini[key]['FP'] += value
                    evaluation_mini[keys]['FN'] += value
                    
    
    dic_update = {}
    for keys,values in evaluation_mini.items(): 
        if (values['TP']+values['FP'])!=0:
            precision = values['TP']/(values['TP']+values['FP'])
        else:
            precision = 0
        if (values['TP']+values['FN'])!=0:
            recall = values['TP']/(values['TP']+values['FN'])
        else:
            recall = 0
    
        dic = {keys:{'precision': precision,
                     'recall':recall, 
                     'f1':2*precision*recall/(precision+recall) if (precision+recall) !=0 else 0}}
        
        dic_update.update(dic)  
        
    return dic_update

def provide_confusion_matrix(evaluation_counts):
    '''
    Read in the evaluation counts and provide a confusion matrix for each class
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :prints out a confusion matrix
    '''
    """
    Compute precision, recall, and F1-score by comparing two paired lists of: system decisions and gold data decisions.
    """
    
    confusion_matrix = pd.DataFrame.from_dict(evaluation_counts)
    confusion_matrix = confusion_matrix.fillna(0)

    return confusion_matrix 

def carry_out_evaluation(gold_annotations, systemfile, systemcolumn, delimiter='\t'):
    '''
    Carries out the evaluation process (from input file to calculating relevant scores)
    
    :param gold_annotations: list of gold annotations
    :param systemfile: path to file with system output
    :param systemcolumn: indication of column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')
    
    returns evaluation information for this specific system
    '''
    system_annotations = extract_annotations(systemfile, systemcolumn, delimiter)
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    provide_confusion_matrix(evaluation_counts)
    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)

    
    return evaluation_outcome

def calculate_overall_score(gold_file, systemfile, systemcolumn1,systemcolumn2, delimiter='\t', mode ='normal'):
    '''
    Calculate the overall precision, recall, and f1-score 
    
    :param gold_file:  path to file with gold output
    :param systemfile: path to file with system output
    :param systemcolumn1: indication of gold column with relevant information
    :param systemcolumn2: indication of system column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')
    
    returns the overall precision, recall, and f1-score in a dictionary format 
    '''
    
    gold_annotations = extract_annotations(gold_file, systemcolumn1, delimiter)
    system_annotations = extract_annotations(systemfile, systemcolumn2, delimiter)
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    provide_confusion_matrix(evaluation_counts)
    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)
    
    precisions = []
    recalls = []
    f1 = []
    for keys, values in evaluation_outcome.items():
        precision = values['precision']
        recall = values['recall']
        f1_score = values['f1']
        precisions.append(precision)
        recalls.append(recall)
        f1.append(f1_score)
    if mode == 'normal':
        p_score = sum(precisions)/5
        r_score = sum(recalls)/5
        f1_score = sum(f1)/5
    else:
        p_score = sum(precisions)/9
        r_score = sum(recalls)/9
        f1_score = sum(f1)/9
    overall_score = {'precision':p_score, 'recall':r_score, 'f1_score':f1_score}
    return overall_score

def provide_output_tables(evaluations):
    '''
    Create tables based on the evaluation of various systems
    
    :param evaluations: the outcome of evaluating one or more systems
    '''
  
    evaluations_pddf = pd.DataFrame.from_dict({(i,j): evaluations[i][j]
                                              for i in evaluations.keys()
                                              for j in evaluations[i].keys()},
                                             orient='index')
    print(evaluations_pddf)
    print(evaluations_pddf.to_latex())
    
def run_evaluations(goldfile, goldcolumn, systems):
    '''
    Carry out standard evaluation for one or more system outputs

    :param goldfile: path to file with goldstandard
    :param goldcolumn: indicator of column in gold file where gold labels can be found
    :param systems: required information to find and process system output
    :type goldfile: string
    :type goldcolumn: integer
    :type systems: list (providing file name, information on tab with system output and system name for each element)

    :returns the evaluations for all systems
    '''
    evaluations = {}
    #not specifying delimiters here, since it corresponds to the default ('\t')
    gold_annotations = extract_annotations(goldfile, goldcolumn)
    for system in systems:
        sys_evaluation = carry_out_evaluation(gold_annotations, system[0], system[1])
        evaluations[system[2]] = sys_evaluation
    return evaluations

def identify_evaluation_value(system, class_label, value_name, evaluations):
    '''
    Return the outcome of a specific value of the evaluation
    
    :param system: the name of the system
    :param class_label: the name of the class for which the value should be returned
    :param value_name: the name of the score that is returned
    :param evaluations: the overview of evaluations
    
    :returns the requested value
    '''
    return evaluations[system][class_label][value_name]

def create_system_information(system_information):
    '''
    Takes system information in the form that it is passed on through sys.argv or via a settingsfile
    and returns a list of elements specifying all the needed information on each system output file to carry out the evaluation.
    
    :param system_information is the input as from a commandline or an input file
    '''
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    systems_list = [system_information[i:i + 3] for i in range(0, len(system_information), 3)]
    return systems_list

def main(my_args=None, mode= 'system1'):
    '''
    A main function. This does not make sense for a notebook, but it is here as an example.
    sys.argv is a very lightweight way of passing arguments from the commandline to a script.
    
    :param mode: you can change the model to different system to test the result. 
    
    '''
    if my_args is None:
        my_args = sys.argv
      
    
    system_info = create_system_information(my_args[2:])
    evaluations = run_evaluations(my_args[0], my_args[1], system_info)
    provide_output_tables(evaluations)
    if mode == 'system1':
        check_eval = identify_evaluation_value('system1', 'ORG', 'precision', evaluations)
        print(check_eval)
    elif mode == 'system2':
        check_eval = identify_evaluation_value('system2', 'O', 'precision', evaluations)
        print(check_eval)
    elif mode == 'system3':
        check_eval = identify_evaluation_value('system3', 'O', 'precision', evaluations)
        print(check_eval)
    elif mode == 'system4':
        check_eval = identify_evaluation_value('system4', 'O', 'precision', evaluations)
        print(check_eval)
    elif mode == 'system5':
        check_eval = identify_evaluation_value('system5', 'O', 'precision', evaluations)
        print(check_eval)
    elif mode == 'system6':
        check_eval = identify_evaluation_value('system6', 'O', 'precision', evaluations)
        print(check_eval)
    elif mode == 'system7':
        check_eval = identify_evaluation_value('system7', 'O', 'precision', evaluations)
        print(check_eval)


# from conll files 
gold_log = 'data/conll2003.dev-preprocessed.predictlogistic'
gold_svm = 'data/conll2003.dev-preprocessed.predictsvm'
gold_nb = 'data/conll2003.dev-preprocessed.predictnb'
gold_w2vtoken = 'data/conll2003.dev-preprocessed.predict_w2vtoken'
gold_w2vpretoken = 'data/conll2003.dev-preprocessed.predict_w2vpretoken'
gold_mix = 'data/conll2003.dev-preprocessed.predict_w2vmix'
gold_crf = 'data/conll2003.dev.conll_revised'
predict_crf ='data/conll2003.dev.crfconll_revised'  

# these can come from the commandline using sys.argv for instance
my_args_1 = [ gold_log ,'gold', gold_log,'predict','system1']
my_args_2 = [ gold_svm,'gold',gold_svm,'predict','system2']
my_args_3 = [gold_nb ,'gold',gold_nb ,'predict','system3']
my_args_4 = [gold_w2vtoken ,'gold',gold_w2vtoken ,'predict','system4']
my_args_5 = [gold_w2vpretoken ,'gold',gold_w2vpretoken ,'predict','system5']
my_args_6 = [gold_mix ,'gold',gold_mix ,'predict','system6']
my_args_7 = [gold_crf ,'gold',predict_crf ,'predict','system7']


main1 = main(my_args_1, mode= 'system1')
overall_score = calculate_overall_score(gold_log, gold_log, 'gold','predict')
print(overall_score,'\n')
main2 = main(my_args_2, mode= 'system2')

overall_score = calculate_overall_score(gold_svm, gold_svm, 'gold','predict')
print(overall_score,'\n')

main3 = main(my_args_3, mode= 'system3')
overall_score = calculate_overall_score(gold_nb, gold_nb, 'gold','predict')
print(overall_score,'\n')

main4 = main(my_args_4, mode= 'system4')
overall_score = calculate_overall_score(gold_w2vtoken, gold_w2vtoken, 'gold','predict')
print(overall_score,'\n')

main5 = main(my_args_5, mode= 'system5')
overall_score = calculate_overall_score(gold_w2vpretoken, gold_w2vpretoken, 'gold','predict')
print(overall_score,'\n')

main6 = main(my_args_6, mode= 'system6')
overall_score = calculate_overall_score(gold_mix, gold_mix, 'gold','predict')
print(overall_score,'\n')

main7 = main(my_args_7, mode= 'system7')
overall_score = calculate_overall_score(gold_crf, predict_crf, 'gold','predict', mode='crf')
print(overall_score,'\n')

# print out confusion matrix from system 1 to 6

goldreal_log = extract_annotations(gold_log,'gold', delimiter='\t')
goldpredict_log = extract_annotations(gold_log, 'predict', delimiter='\t')

goldreal_nb = extract_annotations(gold_nb,'gold', delimiter='\t')
goldpredict_nb = extract_annotations(gold_nb, 'predict', delimiter='\t')

goldreal_svm = extract_annotations(gold_svm,'gold', delimiter='\t')
goldpredict_svm = extract_annotations(gold_svm, 'predict', delimiter='\t')

goldreal_w2vtoken = extract_annotations(gold_w2vtoken,'gold', delimiter='\t')
goldpredict_w2vtoken = extract_annotations(gold_w2vtoken, 'predict', delimiter='\t')

goldreal_w2vpretoken = extract_annotations(gold_w2vpretoken,'gold', delimiter='\t')
goldpredict_w2vpretoken = extract_annotations(gold_w2vpretoken, 'predict', delimiter='\t')

goldreal_w2vmix = extract_annotations(gold_mix,'gold', delimiter='\t')
goldpredict_w2vmix = extract_annotations(gold_mix, 'predict', delimiter='\t')


system1 = obtain_counts(goldreal_log, goldpredict_log)
system2 = obtain_counts(goldreal_nb, goldpredict_nb)
system3 = obtain_counts(goldreal_svm, goldpredict_svm)
system4 = obtain_counts(goldreal_w2vtoken, goldpredict_w2vtoken)
system5 = obtain_counts(goldreal_w2vpretoken, goldpredict_w2vpretoken)
system6 = obtain_counts(goldreal_w2vmix, goldpredict_w2vmix)

print('system1')
print(provide_confusion_matrix(system1),'\n')
print('system2')
print(provide_confusion_matrix(system2),'\n')
print('system3')
print(provide_confusion_matrix(system3),'\n')
print('system4')
print(provide_confusion_matrix(system4),'\n')
print('system5')
print(provide_confusion_matrix(system5),'\n')
print('system6')
print(provide_confusion_matrix(system6),'\n')