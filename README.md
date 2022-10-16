# Machine Learning for NLP: Named Entity Recognition 
Individual Assignment for the Course Machine Learning for NLP at VU University Amsterdam

## AUTHOR
------------------
S.Shen (Shuyi Shen)     

## PROJECT STRUCTURE
-------------------
In this project, the **Portfolio_code** folder contains 4 py files, 4 equivalent jupyter notebooks (in case the py files don't work)
: 
- preprocessing_conll_2021-new 
- CRF
- Train_features_ablation_analysis
- basic_evaluation 

It also includes the **data** folder that contains 2 conll2003 files: 
- train conll2003
- dev conll2003

We have to run the py files or notebooks in the following order:  

- preprocessing_conll_2021-new
- CRF
- Train_features_ablation_analysis
- basic_evaluation

All the packages we need are included in each notebook. 

We specify all the relative path to our inputfile (conll2003 train / dev) and preprocessed outpufile
as 'data/.." in the functions across all the notebooks. 

**Before we run the files, please first download the pre-trained word embedding model 
'GoogleNews-vectors-negative300.bin' and add our given 'conll2003 train' and 'conll2003 dev' to the 
right directory as mentioned - 'data/.." **

`the link to word embedding model- 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'`

### IMPLEMENTED FEATURES

- token itself
- previous token
- next token
- pos tag itself
- previous pos tag
- next pos tag
- Capital
- Chunk labels

### HOW TO USE IT
------------------------------------------------------------------------------------------------------------------------
- STEP 1. preprocessing_conll_2021-new:

We first run this py file or notebook to preprocess the original conll2003 train and dev. 
After running the notebook, we will get two preprocessed files: conll2003 train-preprocessed 
and conll2003 dev-preprocessed, which contain extensive features in additional columns and 
NER annotations without "BIO" scheme, as well as a header. 

------------------------------------------------------------------------------------------------------------------------

- STEP 2.CRF:

Second, we run CRF py file or notebook to predict NER annotations using CRF model with the original conll2003 train and dev. 
We added a narrow_file function to this CRF notebook. After running this notebook, we will get two outputfiles:  
dev.conll_revised (original dev.conll with header added) and dev.crfconll_revised. 
We can load these two files to the 'basic_evaluation' py file / notebook to get the testing result. 

------------------------------------------------------------------------------------------------------------------------

- STEP 3.Train_features_ablation_analysis:

Next, we run features_ablation_analysis py file / notebook to get the outputfile of predicted outcomes of 6 different systems.
The inputdata we use here are "conll2003 train-preprocessed " and "conll2003 dev-preprocessed", both of which we get
from "preprocessing_conll_2021-new" notebook. 
We will load in these outputfiles to the 'basic_evaluation' notebook to get the testing results.
Since this file will create 6 outputfiles (6 systems) using embedding models and traditional features with 
3 different classifiers, this procedure will take much longer than the other 3 py files / notebooks. 

------------------------------------------------------------------------------------------------------------------------
- STEP4:   

Finally, we can run "basic_evaluation" py file or notebook to print out precision, recall, and f1 score for each NER labels with  
confusion matric. In this file, we use pandas function to read in the aforementioned outputfiles. We further use 
zip function and default dict to calculate the evaluation score and confusion matric. We do not use external module, such as  
sklearn to calculate the score in this notebook. We can simply run the code and scroll down to the bottom to see the overall result.  



