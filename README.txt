Assignment for Natural Language Processing course. 

Using a social media corpus consisting of posts from various political sub-Reddits, performed pre-processing, feature extraction, and classification of the data into predicted political leanings (Left, Center, Right, Alt).

a1_preproc.py - Reads input JSON file containing data from the various subreddits. Perform pre-processing. Includes format handling, token POS tagging, and lemmatization. Stores the result as a JSON file.
a1_extractFeatures.py - Extract features based on verbs, tenses, norms (Bristol, Gilhooly, and Logie; Warringer), and LIWC/Receptiviti features. Store the result as a compressed npz file.
a1_classify.py - Run classification experiments using SVC, RandomForest, MLPC, AdaBoost. Tests with parameters, data sizes, K-Fold cross validation and feature values

Your task is to split posts into sentences, tag them with a PoS tagger that we will provide, gather some
feature information from each post, learn models, and use these to classify political persuasion