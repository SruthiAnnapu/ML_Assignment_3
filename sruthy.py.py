import pandas as pd
import time
import nltk
nltk.download('wordnet')
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import timeit
import os

#load data
print("Tweets Data loading")
data_train = pd.read_csv(os.path.join(os.path.dirname(__file__), "train.csv"))

print('Dropping all the columns other than text and target so as to remove noise in the data')

data_train = data_train[['text','target']]

#Eplacing emogies with appropriate characters or words
EMOJIS = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# Removing HTML or ULR tags in the text    
URLPATTERN        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
USERPATTERN       = '@[^\s]+'
SEQPATTERN   = r"(.)\1\1+"
SEQREPLACE = r"\1\1"


contractions = {"ain't": "am not / are not / is not / has not / have not", "aren't": "are not / am not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he had / he would", "he'd've": "he would have", "he'll": "he shall / he will", "he'll've": "he shall have / he will have", "he's": "he has / he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how has / how is / how does", "I'd": "I had / I would", "I'd've": "I would have", "I'll": "I shall / I will", "I'll've": "I shall have / I will have", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it had / it would", "it'd've": "it would have", "it'll": "it shall / it will", "it'll've": "it shall have / it will have", "it's": "it has / it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she had / she would", "she'd've": "she would have", "she'll": "she shall / she will", "she'll've": "she shall have / she will have", "she's": "she has / she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as / so is", "that'd": "that would / that had", "that'd've": "that would have", "that's": "that has / that is", "there'd": "there had / there would", "there'd've": "there would have", "there's": "there has / there is", "they'd": "they had / they would", "they'd've": "they would have", "they'll": "they shall / they will", "they'll've": "they shall have / they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we had / we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what shall / what will", "what'll've": "what shall have / what will have", "what're": "what are", "what's": "what has / what is", "what've": "what have", "when's": "when has / when is", "when've": "when have", "where'd": "where did", "where's": "where has / where is", "where've": "where have", "who'll": "who shall / who will", "who'll've": "who shall have / who will have", "who's": "who has / who is", "who've": "who have", "why's": "why has / why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you had / you would", "you'd've": "you would have", "you'll": "you shall / you will", "you'll've": "you shall have / you will have", "you're": "you are", "you've": "you have"}
contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
def expand_contractions(s, contractions = contractions):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, s)

#method to normalize data, remove URL, replace EMOJI's and remove stopwords in the data
def cleandata(df):
    wordLemm = WordNetLemmatizer()
    print("Cleaning and normalising the data")
    t = time.time()
    corpus = []
    # Expand_contractions
    data_train['text'] = data_train['text'].apply(expand_contractions)
    for i in range(0, len(df['text'])):
        ## lower casing
        text = df["text"][i].lower()
        ### Replacing URL
        text = re.sub(URLPATTERN, ' URL', text)
        ### Replacing EMOJI
        for emoji in EMOJIS.keys():
            text = text.replace(emoji, "EMOJI" + EMOJIS[emoji])
        ### Replacing USER pattern
        text = re.sub(USERPATTERN, ' URL', text)

        ### Removing non-alphabets
        text = re.sub('[^a-zA-z]', " ", text)

        ###Removing html tags
        text = re.compile(r'<.*?>').sub(r'',text)

        ### Removing consecutive letters
        text = re.sub(SEQPATTERN, SEQREPLACE, text)
        text = text.split()

        # Removing consecutive letters and stopwords and Word Lemmatizing
        text = [wordLemm.lemmatize(word) for word in text if not word in stopwords.words('english') and len(word) > 1]
        text = ' '.join(text)
        corpus.append(text)

    return corpus

def RandomForest(X_train, y_train, X_test, y_test):
    print("Initiated Random Forest Classifer")
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return model, cm, score, f1

def LogisticRegressionClassifier(X_train, y_train, X_test, y_test):
    print("Initiated Logistic Regression Classifer")
    lr = LogisticRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return lr, cm, score, f1

def NaiveBayes(X_train, y_train, X_test, y_test):
    print("Initiated Naive Bayes Classifer")
    model = MultinomialNB().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return model, cm, score, f1

corpus = cleandata(data_train)
tfid = TfidfVectorizer(max_features=5000,ngram_range=(1,2))
X_tfid = tfid.fit_transform(corpus).toarray()
# print("printing the shape")
# print(X_tfid.columns)
y = data_train['target']

X_train, X_test, y_train, y_test = train_test_split(X_tfid, y, test_size=0.20, random_state=0)
#X_train = data_train[['text','target']]
#X_test = data_test[['text']]
#y_train = data_train['text','target']
#y_test = data_test['text']
#start = []
#end = []

print(y_train.shape)
print(y_test.shape)


start = time.time()

rf_model, rf_cm, rf_score, rf_f1 = RandomForest(X_train, y_train, X_test, y_test)
end = time.time()
print("Accuracy for Random Forest Classifier: "+str(rf_score))
print("F1-score for Random Forest Classifier: "+str(rf_f1))
print("Time taken by RF: "+str(end - start))
print(rf_cm)
start = end
lr_model, lr_cm, lr_score, rf_lr = LogisticRegressionClassifier(X_train, y_train, X_test, y_test)
end = time.time()
print("Accuracy for Logistic Regression Classifier: "+str(lr_score))
print("F1-score for Random Logestic Regression: "+str(rf_lr))
print("Time taken by lr: "+str(end - start))
print(lr_cm)

start = end
nb_model, nb_cm, nb_score, rf_nb = NaiveBayes(X_train, y_train, X_test, y_test)
end = time.time()
print("Accuracy for Naive Bayes Classifier: "+str(nb_score))
print("F1-score for Random Naive Bayes Classifier: "+str(rf_nb))
print("Time taken by NB: "+str(end - start))
print(nb_cm)


