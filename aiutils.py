from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import joblib
import json

model = joblib.load('./model/my_model_for_healthcare.joblib')
prognosis = json.load(open('./model/prognosis.json'))
VOCAB_DB = [' '.join(x.split()) for x in prognosis]
MAX_NGRAM_LEN = max([len(x.split()) for x in VOCAB_DB])

def vectorize(symptoms):
    found_symptoms = [int(x in symptoms) for x in prognosis]
    return found_symptoms

def extract_terms(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.isalpha()]
    vectorizer = CountVectorizer(ngram_range=(1, MAX_NGRAM_LEN), vocabulary=VOCAB_DB)
    filtered_str = ' '.join(filtered_tokens)
    X = vectorizer.transform([filtered_str])
    clf = MultinomialNB()
    clf.fit(X, ['health-related'])
    prediction = clf.predict(X)
    if prediction[0] == 'health-related':
        return [term.replace(' ','_') for term in vectorizer.get_feature_names_out() if term in filtered_str]
    return []


def diagnosis(data):
    et = extract_terms(data)
    v = vectorize(et)
    print('yd', v)
    y_diagnosis = model.predict([v])
    y_pred_2 = model.predict_proba([v])
    report = {}
    report['disease'] = y_diagnosis[:2]
    report['confidence'] = y_pred_2.max()*100
    return report

# # Sample narrative text
# text = "I've been experiencing high temperature, severe headache, skin rash, and fatigue for the past few days. I also have a high fever."
# # text = input("Enter: ").strip()
# # print(extract_terms(text))
# et = extract_terms(text)
# v = vectorize(et)
# print(et)
# print(v)

# # a = vectorize(['skin_rash','pain_behind_the_eyes', 'back_pain'])

# y_diagnosis = model.predict([v])
# y_pred_2 = model.predict_proba([v])
# print(('Name of the infection = %s , confidence score of : = %s') %(y_diagnosis[0],y_pred_2.max()* 100),'%' )