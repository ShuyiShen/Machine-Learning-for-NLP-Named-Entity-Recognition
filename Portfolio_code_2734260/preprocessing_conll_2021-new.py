import csv
from typing import List, Dict
from collections import deque

def write_feature(inputdata, outputfile):
    """
    This function helps you extend the original features and 
    rmeove the prefix of NER annotations, converitng them into 
    the same format. The result returns a preprocessed csv file 
    that contains additional 5 features, including preceding, next
    tokens and pos tags and capitals. 
    
    :param inputdata: training file 
    :param outputfile: testing file 
    
    """
    # we created lists to append the additional features
    tokens = []
    caps = [] # capitals 
    pos = [] # pos tags 
    simple_ner = [] # NER annotations without BIO format 
    
    #  We load in training (conll2003 train) or testing file (conll2003 dev)
    # and rotate the deque object to extract the preceding and next tokens and pos tags 
    # we removed the prefix of NER labels, and if the row in the training or testing file 
    # is empty, we append "ENDSENTX" to avoid the index error when writing in the file. 
    
    with open(inputdata, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            
            if len(components) > 0:
                token = components[0]
                pos_tag = components[1]
                pos.append(pos_tag)
                tokens.append(token)
                upper = str(token[0].isupper())
                caps.append(upper)
                BIOner = components[3]
                if BIOner[0] == 'B' or BIOner[0] == 'I':
                    sn = BIOner[2:]
                else:
                    sn = BIOner
                simple_ner.append(sn)
                
            else:
                tokens.append("ENDSENTX")
                caps.append("ENDSENTX")
                pos.append("ENDSENTX")
                simple_ner.append("ENDSENTX")
                
                
    # we print out the lenth of additional features to double check if the overall number of our features is correct. 
    
    print(len(tokens))

    prev = deque(tokens)
    prev.rotate(1)
    prev_tokens = list(prev)

    print(len(prev_tokens))

    prev_ = deque(pos)
    prev.rotate(1)
    prev_pos = list(prev_)

    print(len(prev_pos))

    next_ = deque(tokens)
    next_.rotate(-1)
    next_tokens = list(next_)

    print(len(next_tokens))

    next_pos = deque(pos)
    next_pos.rotate(-1)
    next_pos = list(next_pos)

    print(len(next_pos))
    
    
    # we write in the additional features that were created above
    # we also write in the header for the convenience to extract the column of a specific feature using pandas 
    
    with open (outputfile, 'w', newline='', encoding = 'utf-8') as outfile:
        count = 0
        for line, prev_token, next_token, prevpos, nextpos, cap, NER in zip(open(inputdata, 'r'), prev_tokens, next_tokens, prev_pos, next_pos, caps, simple_ner):
            
            if count == 0:
                
                outfile.write('token' + '\t' + 'pos' + '\t' + 'chunk' + '\t' + 'ner'+'\t'+'prev_token'+'\t'+'next_token'+'\t'+'prevpos'+'\t'+'nextpos'+'\t'+'capital'+ '\t'+'NER'+'\n')
                outfile.write(line.rstrip('\n') + '\t' + prev_token + '\t' + next_token + '\t' + prevpos+'\t'+nextpos+'\t'+cap + '\t'+NER+'\n')    
                    
            else:
                
                if len(line.rstrip('\n').split()) > 0:        
                    outfile.write(line.rstrip('\n') + '\t' + prev_token + '\t' + next_token + '\t' + prevpos+'\t'+nextpos+'\t'+cap +'\t' + NER + '\n')
                else:
                    
                    pass_ = ("O"+'\t')*9 
                    outfile.write(pass_+'O'+'\n')
                    
            count +=1
            
inputdata = 'data/conll2003.train.conll'
outputfile = 'data/conll2003.train-preprocessed.conll'

write_feature(inputdata, outputfile)

inputdata = 'data/conll2003.dev.conll'
outputfile = 'data/conll2003.dev-preprocessed.conll'

write_feature(inputdata, outputfile)