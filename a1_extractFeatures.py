import numpy as np
import sys
import argparse
import os
import json
import re

BGL_csv_filename = '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
Warringer_csv_filename = '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'

# Load the BGL file into a dictionary for easy lookup later
BGL_dict = {}
f = open(BGL_csv_filename, 'r')
count = 0
header_flag = True
headers = []
for row in f:
    if header_flag:
        header_flag = False
        headers = row.strip()
        headers = headers.split(',')
        continue
    temp_array = row.strip()
    temp_array = temp_array.split(',')
    if (temp_array[1] != '' and temp_array[1] not in BGL_dict):
        BGL_dict[temp_array[1]] = {
            headers[3]: temp_array[3],
            headers[4]: temp_array[4],
            headers[5]: temp_array[5]
        }
        
# Load the Warringer file into a dictionary for easy lookup later
Warringer_dict = {}
f = open(Warringer_csv_filename, 'r')
count = 0
header_flag = True
headers = []
for row in f:
    if header_flag:
        header_flag = False
        headers = row.strip()
        headers = headers.split(',')
        continue
    temp_array = row.strip()
    temp_array = temp_array.split(',')
    if (temp_array[1] != '' and temp_array[1] not in Warringer_dict):
        Warringer_dict[temp_array[1]] = {
            headers[2]: temp_array[2],
            headers[5]: temp_array[5],
            headers[8]: temp_array[8]
        }

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    
    featuresArray = np.zeros((1, 173))
    
    # 1-3: Number of pronouns
    # List of pronouns in the first, second, and third person
    first_person_pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    second_person_pronouns = ['you', 'your', 'yours', 'u', 'ur', 'urs']
    third_person_pronouns = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'their']
    first_person_pronouns_count = 0
    second_person_pronouns_count = 0
    third_person_pronouns_count = 0
    
    # Search the above pronouns, using the /PRP and /PRP$ tags
    pattern = re.compile('(\w+)\/PRP[$]?')
    searched = pattern.findall(comment)
    
    #1.  Number of first-person pronouns => First person: I, me, my, mine, we, us, our, ours
    #2.  Number of second-person pronouns => Second person: you, your, yours, u, ur, urs
    #3.  Number of third-person pronouns => Third person: he, him, his, she, her, hers, it, its, they, them, their, their
    if (searched == []):
        # If no pronouns were found in the comment
        featuresArray[0][:3] = [0, 0, 0]
    else:
        # If pronouns were found
        for pronoun in searched:
            if pronoun in first_person_pronouns: first_person_pronouns_count +=  1
            elif pronoun in second_person_pronouns: second_person_pronouns_count +=  1
            elif pronoun in third_person_pronouns: third_person_pronouns_count +=  1
        featuresArray[0][:3] = [first_person_pronouns_count, second_person_pronouns_count, third_person_pronouns_count]
    

    #4.  Number of coordinating conjunctions
    pattern = re.compile('(\w+)\/CC')
    searched = pattern.findall(comment)
    featuresArray[0][3] = len(searched)

    #5.  Number of past-tense verbs
    pattern = re.compile('(\w+)\/VB[DN]')
    searched = pattern.findall(comment)
    featuresArray[0][3] = len(searched)

    #6.  Number of future-tense verbs
    # ’ll, will, gonna, going to + VB
    # This seems like it will be almost entirely negated by the removal of StopWords
    pattern = re.compile('((\'ll\/MD\w*|will\/MD\w*|gonna\/\w+)\s+\w+\/VB)')
    pattern2 = re.compile('(go\/VB\w*\s+to\/TO\w*\s+\w+\/VB)')
    searched = pattern.findall(comment)
    searched2 = pattern2.findall(comment)
    featuresArray[0][5] = len(searched) + len(searched2)

    #7.  Number of commas
    pattern = re.compile('\S+/,')
    searched = pattern.findall(comment)
    featuresArray[0][6] = len(searched)

    #8.  Number of multi-character punctuation tokens
    pattern = re.compile('([?!,;:\.\-`"]{2,})\/')
    searched = pattern.findall(comment)
    featuresArray[0][7] = len(searched)

    #9.  Number of common nouns
    # NN, NNS
    pattern = re.compile('(\w+)\/NNS?')
    searched = pattern.findall(comment)
    featuresArray[0][8] = len(searched)

    #10.  Number of proper nouns
    # NNP, NNPS
    pattern = re.compile('(\w+)\/NNPS?')
    searched = pattern.findall(comment)
    featuresArray[0][9] = len(searched)

    #11.  Number of adverbs
    # RB, RBR, RBS
    pattern = re.compile('(\w+)\/RB[RS]?')
    searched = pattern.findall(comment)
    featuresArray[0][10] = len(searched)

    #12.  Number of wh- words
    # WDT, WP, WP$, WRB
    pattern = re.compile('(\w+)\/W(DT|P|P$|RB)')
    searched = pattern.findall(comment)
    featuresArray[0][11] = len(searched)

    #13.  Number of slang acronyms
    # Not all, but all the common ones from the list:
    #   smh,  fwb,  lmfao,  lmao,  lms,  tbh,  rofl,  wtf,  bff,  wyd,  lylc,  brb,  atm,  imao,  sml,  btw,
    #   bw, imho, fyi, ppl, sob, ttyl, imo, ltr, thx, kk, omg, omfg, ttys, afn, bbs, cya, ez, f2f,
    #   gtr,  ic,  jk,  k,  ly,  ya,  nm,  np,  plz,  ru,  so,  tc,  tmi,  ym,  ur,  u,  sol,  fml
    pattern = re.compile('\s(smh|fwb|lmfao|lmao|lms|tbh|rofl|wtf|bff|wyd|lylc|brb|atm|imao|sml|btw|bw|imho|fyi|ppl|sob|ttyl|imo|ltr|thx|kk|omg|omfg|ttys|afn|bbs|cya|ez|f2f|gtr|ic|jk|k|ly|ya|nm|np|plz|ru|so|tc|tmi|ym|ur|u|sol|fml)\/\S+')
    searched = pattern.findall(comment)
    featuresArray[0][12] = len(searched)

    #14.  Number of words in uppercase (≥ 3 letters long)
    # This should never be triggered because everything is put in lowercase...
    pattern = re.compile('[A-Z]{3,}\/\S+')
    searched = pattern.findall(comment)
    featuresArray[0][13] = len(searched)


    # 15-17: Pre-computing number of sentences, tokens/sentence, and average token length
    temp_comment = comment.rstrip('\n')
    sentence_array = temp_comment.split('\n')
    num_tokens = 0
    token_sum = 0
    for sentence in sentence_array:
        pattern = re.compile('\S+\/\S+')
        tokens = pattern.findall(sentence)
        num_tokens += len(tokens)
        for token in tokens:
            token_sum += len(str(token))
        
    #15.  Average length of sentences, in tokens
    if len(sentence_array) == 0:
        featuresArray[0][14] = 0
    else:
        featuresArray[0][14] = num_tokens/len(sentence_array)

    #16.  Average length of tokens, excluding punctuation-only tokens, in characters
    if num_tokens == 0:
        featuresArray[0][15] = 0
    else:    
        featuresArray[0][15] = token_sum/num_tokens

    #17.  Number of sentences
    featuresArray[0][16] = len(sentence_array)
       
    
    
    # Bristol, Gilhooly, and Logie norms CSV is pulled from '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
    # Handled globally, and converted to dictionaries for easy use here in parts 18-23
    # With computation below, words that are not in these tables will inherit the average value of the other words in the comment
    AoA_sum = 0
    IMG_sum = 0
    FAM_sum = 0
    valid_word_count = 0
    pattern = re.compile('(\S+)\/\S+')
    searched = pattern.findall(comment)
    for word in searched:
        if word in BGL_dict and word != '':
            valid_word_count += 1
            AoA_sum += float(BGL_dict[word]['AoA (100-700)'])
            IMG_sum += float(BGL_dict[word]['IMG'])
            FAM_sum += float(BGL_dict[word]['FAM'])
    
    #18.  Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    #19.  Average of IMG from Bristol, Gilhooly, and Logie norms
    #20.  Average of FAM from Bristol, Gilhooly, and Logie norms
    if valid_word_count == 0:
        # Avoid divide by zero case
        featuresArray[0][17:23] = [0,0,0,0,0,0]
    else:
        # Compute Average
        featuresArray[0][17] = AoA_sum/valid_word_count
        featuresArray[0][18] = IMG_sum/valid_word_count
        featuresArray[0][19] = FAM_sum/valid_word_count
        
        #21.  Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
        #22.  Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
        #23.  Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
        AoA_SD = 0
        IMG_SD = 0
        FAM_SD = 0
        for word in searched:
            if word in BGL_dict and word != '':
                # Same as above, but for Standard Deviation calculation. This is numerator term
                AoA_SD += (float(BGL_dict[word]['AoA (100-700)']) - float(featuresArray[0][17]))**2
                IMG_SD += (float(BGL_dict[word]['IMG']) - float(featuresArray[0][18]))**2
                FAM_SD += (float(BGL_dict[word]['FAM']) - float(featuresArray[0][19]))**2
        
        # Compute Standard Deviation
        featuresArray[0][20] = AoA_SD/valid_word_count
        featuresArray[0][21] = IMG_SD/valid_word_count
        featuresArray[0][22] = FAM_SD/valid_word_count
        
        
    # Warringer norms CSV is pulled from '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'
    # Handled globally, and converted to dictionaries for easy use here in parts 24-29
    # With computation below, words that are not in these tables will inherit the average value of the other words in the comment
    V_sum = 0
    A_sum = 0
    D_sum = 0
    valid_word_count = 0
    pattern = re.compile('(\S+)\/\S+')
    searched = pattern.findall(comment)
    for word in searched:
        if word in Warringer_dict and word != '':
            valid_word_count += 1
            V_sum += float(Warringer_dict[word]['V.Mean.Sum'])
            A_sum += float(Warringer_dict[word]['A.Mean.Sum'])
            D_sum += float(Warringer_dict[word]['D.Mean.Sum'])
    
    #24.  Average of V.Mean.Sum from Warringer norms
    #25.  Average of A.Mean.Sum from Warringer norms
    #26.  Average of D.Mean.Sum from Warringer norms
    if valid_word_count == 0:
        # Avoid divide by zero case
        featuresArray[0][23:29] = [0,0,0,0,0,0]
    else:
        # Compute Average
        featuresArray[0][23] = V_sum/valid_word_count
        featuresArray[0][24] = A_sum/valid_word_count
        featuresArray[0][25] = D_sum/valid_word_count
        
        #27.  Standard deviation of V.Mean.Sum from Warringer norms
        #28.  Standard deviation of A.Mean.Sum from Warringer norms
        #29.  Standard deviation of D.Mean.Sum from Warringer norms
        V_SD = 0
        A_SD = 0
        D_SD = 0
        for word in searched:
            if word in Warringer_dict and word != '':
                # Same as above, but for Standard Deviation calculation. This is numerator term
                V_SD += (float(Warringer_dict[word]['V.Mean.Sum']) - float(featuresArray[0][23]))**2
                A_SD += (float(Warringer_dict[word]['A.Mean.Sum']) - float(featuresArray[0][24]))**2
                D_SD += (float(Warringer_dict[word]['D.Mean.Sum']) - float(featuresArray[0][25]))**2
        
        # Compute Standard Deviation
        featuresArray[0][26] = V_SD/valid_word_count
        featuresArray[0][27] = A_SD/valid_word_count
        featuresArray[0][28] = D_SD/valid_word_count
    
    return featuresArray


def main( args ):
    
    feats_location = '/u/cs401/A1/feats/'
    Alt_feats_file_name = 'Alt_feats.dat.npy'
    Center_feats_file_name = 'Center_feats.dat.npy'
    Right_feats_file_name = 'Right_feats.dat.npy'
    Left_feats_file_name = 'Left_feats.dat.npy'
    Alt_IDs_file_name = 'Alt_IDs.txt'
    Center_IDs_file_name = 'Center_IDs.txt'
    Right_IDs_file_name = 'Right_IDs.txt'
    Left_IDs_file_name = 'Left_IDs.txt'
    
    # Load npy feats files
    Left_feats_file = np.load(feats_location + '/' + Left_feats_file_name)
    Center_feats_file = np.load(feats_location + '/' + Center_feats_file_name)
    Right_feats_file = np.load(feats_location + '/' + Right_feats_file_name)
    Alt_feats_file = np.load(feats_location + '/' + Alt_feats_file_name)
    
    # Load ID files
    left_f = open(feats_location + '/' + Left_IDs_file_name, 'r').read()
    Left_IDs_array = left_f.split('\n')
    center_f = open(feats_location + '/' + Center_IDs_file_name, 'r').read()
    Center_IDs_array = center_f.split('\n')
    right_f = open(feats_location + '/' + Right_IDs_file_name, 'r').read()
    Right_IDs_array = right_f.split('\n')
    alt_f = open(feats_location + '/' + Left_IDs_file_name, 'r').read()
    Alt_IDs_array = alt_f.split('\n')
    
    # Load inputted data from JSON (output from a1_preproc)
    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    # Provide enumerations for the class as per the cat JSON field
    cat_lookup = {'Left': 0, 'Center': 1, 'Right': 2, 'Alt': 3}
    for i in range(len(feats)):
        
        line = data[i]
        
        # For debug - need to run it with this field inputted from a1_preproc.py
        #print(line['old_body'] + '\n')
        #print("{}/{}".format(i, len(feats)))
        
        # Store the 173-length vector of floating point features as the first 173 columns for each row (comment)
        feats[i][:173] = extract1(line['body'])

        #30-173.  LIWC/Receptiviti features
        if cat_lookup[line['cat']] == 0:
            # Left
            for j in range(len(Left_IDs_array)):
                if line['id'] == Left_IDs_array[j]:
                    feats[i][29:173] = Left_feats_file[j][:]
            
        elif cat_lookup[line['cat']] == 1:
            # Center
            for j in range(len(Center_feats_file)):
                if line['id'] == Center_IDs_array[j]:
                    feats[i][29:173] = Center_feats_file[j][:]
            
        elif cat_lookup[line['cat']] == 2:
            # Right
            for j in range(len(Right_IDs_array)):
                if line['id'] == Right_IDs_array[j]:
                    feats[i][29:173] = Right_feats_file[j][:]
        
        elif cat_lookup[line['cat']] == 3:
            # Alt
            for j in range(len(Alt_IDs_array)):
                if line['id'] == Alt_IDs_array[j]:
                    feats[i][29:173] = Alt_feats_file[j][:]
                    
        # Store the class (as per the cat JSON field) as the last column for each row (comment)
        feats[i][173] = cat_lookup[line['cat']]
    
    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)
