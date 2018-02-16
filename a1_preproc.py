import sys
import argparse
import os
import json
import re
import html
import spacy
 
indir = '/u/cs401/A1/data/';

# Set up variables that will require time-intensive, or permission-granting requirements
# spaCy setup
nlp = spacy.load('en', disable=['parser', 'ner'])

# Reading all StopWords from /u/cs401/Wordlists/StopWords
f = open('/u/cs401/Wordlists/StopWords', 'r')
tempStopWords = []
for line in f:
    tempStopWords.append(line.strip())
f.close()

 
def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment
 
    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  
 
    Returns:
        modComm : string, the modified comment 
    '''
 
    modComm = ''
    if 1 in steps:
        # Remove end line characters
        modComm = re.sub('\n+', ' ', comment)
    if 2 in steps:
        # Replace all HTML character codes with ASCII equivalent
        modComm = html.unescape(modComm)
    if 3 in steps:
        # Remove all URLs (beginning with http(s): or www.
        modComm = re.sub('www\S+', '', modComm)
        modComm = re.sub('https?\S+', '', modComm)
    if 4 in steps:
        # Remove punctuation excpet apostrophes 
        modComm = re.sub('([\d\w]+)(\.[^\d\w]*)', r'\1 \2 ', modComm)             # Get rid of any periods following words/numbers that aren't abbreviations
        modComm = re.sub('([\d\w ]+)([?!,;:\.\-`"]+[^\d\w]*)', r'\1 \2 ', modComm) # Get rid of any strings of punctuation following words/numbers that aren't abbreviations
        modComm = re.sub(' +', ' ', modComm)                                      # Remove excess whitespace
    if 5 in steps:
        # Separate cltiics by whitespace: 've, 's, s', 're, 'll, n't, 'd, 'm
        modComm = re.sub('n\'t', ' n\'t', modComm)
        modComm = re.sub('\'ll', ' \'ll', modComm)
        modComm = re.sub('\'d', ' \'d', modComm)
        modComm = re.sub('\'ve', ' \'ve', modComm)
        modComm = re.sub('\'re', ' \'re', modComm)
        modComm = re.sub('\'m', ' \'m', modComm)
        modComm = re.sub('\'s', ' \'s', modComm)
        modComm = re.sub('s\'(\w)', r"s\' \1", modComm)
    if 6 in steps:
        # Use spaCy to tag tokens
        utt = nlp(modComm)
        doc = spacy.tokens.Doc(nlp.vocab, words=modComm.strip().split())
        doc = nlp.tagger(doc)
        modComm = ''
        for token in doc:
            modComm += token.text + '/' + token.tag_ + ' '
    if 7 in steps:
        # Remove StopWords: list was globally pulled from /u/cs401/Wordlists/StopWords
        pattern = re.compile('(\w+\/[A-Z\.?;:,\(\)\'"`$#]+)')
        searched = pattern.findall(modComm)
        for i in searched:
            tempstr = re.sub('(\w+)\/.+', r'\1', i)
            if tempstr.lower() in tempStopWords:
                modComm = re.sub(tempstr + '\/([A-Z\.?;:,\(\)\'"`$#]+)', ' ', modComm)
    if 8 in steps:
        # Use spaCy to replace tokens with token lemmas
        modComm = re.sub('\/([A-Z\.?;:,\(\)\'"`$#]+)', '', modComm)
        utt = nlp(modComm)
        doc = spacy.tokens.Doc(nlp.vocab, words=modComm.strip().split())
        doc = nlp.tagger(doc)
        modComm = ''
        for token in doc:
            # Check for dashes in the -PRON- case
            if (re.match('\-\w+', str(token.lemma_)) != None and re.match('\-\w+', str(token)) == None):
                modComm += str(token) + '/' + str(token.tag_) + ' '
            else:
                modComm += str(token.lemma_) + '/' + str(token.tag_) + ' '
    if 9 in steps:
        # Replace end of sentences with new-line characters
        modComm = re.sub('([\w\d]+\s+[\.?!]\/\.)\s', r'\1\n', modComm)
    if 10 in steps:
        # Conver to lowercase
        modComm = re.sub('(\w+\/)([A-Z\.?;:,\(\)\'"`$#]+)', lambda r: r.group(1).lower() + r.group(2), modComm)
        modComm = re.sub(' +', ' ', modComm)    # Clean up extra whitespaces one more time
        
    return modComm
 
def main( args ):
 
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)
 
            data = json.load(open(fullFile))
            data_len = len(data)
            
            line_count = 0
            index = args.ID[0]%data_len
            # Iterate args.max lines (starting from ID%datalength'th) line in each file
            # Using a while loop and not a for loop, because will be skipping invalid (deleted comments)
            while line_count < args.max:
                
                # Get the json line - use cyclic % check to make sure it stays in bound
                line = data[index%data_len]

                # Read each line
                j = json.loads(line)
                
                # Retain relevant fields: body 
                comment = j['body']
                
                # Ignore any deleted or empty comments
                if not (comment == None or comment == "" or comment == "[deleted]" or comment == "[removed]"):
                    # Iterate line_count iterator
                    line_count += 1
                    
                    # Process the body field with preproc1, using default for `steps` argument
                    # Replace the body field with this processed text
                    j['body'] = preproc1(comment)
                    
                    # Add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                    j['cat'] = file

                    # Print statements to monitor progress. Should be commented out for submission                    
                    #print("index: {}, cat: {}, body: {}".format(index%data_len, j['cat'], j['body']))
                    #print("{} count: {}/{}".format(j['cat'], line_count, args.max))
                    #print("index: {}".format(index%data_len))
                    #print("cat: {}".format(j['cat']))
                    #print("initial body: {}\n".format(comment))
                    #print("modified body: {}".format(j['body']))
                    #print("\n\n")
                    
                    # Append the result to 'allOutput'
                    allOutput.append(j)
                
                # Once we've read args.max lines, break
                if line_count >= args.max: break
                
                # Update iterator to read next line on next cycle
                index += 1
	    
            # DONE: select appropriate args.max lines
            # DONE: read those lines with something like `j = json.loads(line)`
            # DONE: choose to retain fields from those lines that are relevant to you
            # DONE: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
            # DONE: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # DONE: replace the 'body' field with the processed text
            # DONE: append the result to 'allOutput'
             
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()
 
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()
 
    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
         
    main(args)
