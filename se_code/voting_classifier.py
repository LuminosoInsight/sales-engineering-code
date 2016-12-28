from luminoso_api import LuminosoClient
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from pack64 import pack64, unpack64
import numpy as np
import argparse
import math
    
def sigmoid(x):
    '''Map distance to hyperplane to a range 0-1. Approximation of "likelihood" that the prediction is correct.'''
    
    return [[1 / (1 + math.exp(-y)) for y in z] for z in x]

def combine_decision_functions(cls_dfuncs, classes, weights=None):
    '''Combine outputs from multiple classifiers. Apply weights. Take sum of result across classifiers.'''
 
    if classes == 2:
        cls_dfuncs = [[(-b,b) for b in a] for a in cls_dfuncs]
        
    if weights:
        cls_funcs = [c*w for c,w in zip(cls_dfuncs,weights)]
    
    classification = np.dstack(cls_dfuncs)
    
    classification = np.sum(classification, axis=2)
    
    return sigmoid(classification)
  
def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    
    z = x.copy()
    z.update(y)
    return z

def strip_term(term):
    '''Ensure collocations are vectorized together through joining with underscore'''
    
    words = [word[:-3] for word in term.split() if word.endswith('|en')]
    return '_'.join(words)

def sklearn_text(doc):
    '''Return lumi-fied text for vectorization'''
    
    stripped_terms = [strip_term(term) for (term, _, _) in doc['terms']]
    return ' '.join(stripped_terms)

def get_all_docs(client, subset_field, batch_size=20000):
    '''Pull all docs from project'''
    
    docs = []
    offset = 0
    while True:
        newdocs = client.get('docs', offset=offset, limit=batch_size)
        
        if not newdocs:
            labels = [doc['subsets'][i][len(subset_field):] for doc in docs for i,s in enumerate(doc['subsets']) if subset_field in doc['subsets'][i]]
            docs = [doc for doc in docs if any(subset_field in s for s in doc['subsets'])]
            return docs, labels
        
        docs.extend(newdocs)
        offset += batch_size
        
def split_train_test(docs, labels, split=0.3):
    '''Split documents & labels into a single dict each for both testing and training'''
    
    return train_test_split(np.array(docs), np.array(labels), test_size=split, random_state=32, stratify=np.array(labels))
        
def make_term_vectorizer():
    '''Vectorize Luminoso terms'''
    
    # Consider min_df?
    return TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=None, token_pattern=r'\S+')

def make_simple_vectorizer():
    '''Vectorize all words'''
    
    # Consider min_df?
    return TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

def train_classifier(client, subset_field, project_type):
    '''Train classifiers & vectorizers on training data set'''
    
    docs, labels = get_all_docs(client, subset_field)
    
    if project_type == 'combined': 
        docs,_,labels,_ = split_train_test(docs, labels)
    
    term_vectorizer = make_term_vectorizer()
    simple_vectorizer = make_simple_vectorizer()
    vectorizers = {'simple': simple_vectorizer, 'term': term_vectorizer}
    
    classifiers = {
        style: LinearSVC(
            C=1.0, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr',
            penalty='l1', random_state=None, tol=0.0001, verbose=0
        )
        for style in ('simple', 'term', 'vector', 'new_vects')
    }
    
    simple_vecs = vectorizers['simple'].fit_transform([doc['text'] for doc in docs])
    term_vecs = vectorizers['term'].fit_transform([sklearn_text(doc) for doc in docs])
    luminoso_vecs = normalize([unpack64(doc['vector']) for doc in docs])
    
    classifiers['simple'].fit(simple_vecs, labels)
    classifiers['term'].fit(term_vecs, labels)
    classifiers['vector'].fit(luminoso_vecs, labels)
    
    return (classifiers, vectorizers)

def test_classifier(train_client, test_client, classifiers, vectorizers, subset_field, project_type, save_results=False):
    '''Test classification using reserved training set'''
    
    test_docs, labels = get_all_docs(test_client, subset_field)
    if project_type == 'combined':
        _,test_docs,_,_labels = split_train_test(test_docs, labels)
        
    test_docs = train_client.upload('docs/vectors', test_docs)
    simple_vecs = vectorizers['simple'].transform([doc['text'] for doc in test_docs])
    term_vecs = vectorizers['term'].transform([sklearn_text(doc) for doc in test_docs])
    luminoso_vecs = normalize([unpack64(doc['vector']) for doc in test_docs])

    classification = combine_decision_functions([
        classifiers['simple'].decision_function(simple_vecs),
        classifiers['term'].decision_function(term_vecs),
        classifiers['vector'].decision_function(luminoso_vecs)
    ],
            len(classifiers['simple'].classes_))
        
    if save_results:
        import csv
        results_dict = [merge_two_dicts({'text': z[0]['text'],'truth': z[1]},dict(zip(list(classifiers['simple'].classes_),z[2]))) for z in zip(test_docs,labels,classification)]
        writer = csv.DictWriter(open('results.csv','w',encoding='utf-8'),['text','truth']+list(classifiers['simple'].classes_))
        writer.writeheader()
        writer.writerows(results_dict)
        
    return classification

def return_label(new_text, classifiers, vectorizers, train_client):
    '''Return label function for operating in a live demo'''
    
    test_docs = train_client.upload('docs/vectors', [{'text':new_text}])
    simple_vecs = vectorizers['simple'].transform([doc['text'] for doc in test_docs])
    term_vecs = vectorizers['term'].transform([sklearn_text(doc) for doc in test_docs])
    luminoso_vecs = normalize([unpack64(doc['vector']) for doc in test_docs])

    classification = combine_decision_functions([
        classifiers['simple'].decision_function(simple_vecs),
        classifiers['term'].decision_function(term_vecs),
        classifiers['vector'].decision_function(luminoso_vecs)
    ],
            len(classifiers['simple'].classes_))

    best_class = np.argmax(classification, axis=1)[0]
    return classifiers['simple'].classes_[best_class],classification[0][best_class]

def score_results(test_client, classifiers, classification, subset_field, project_type):
    '''Return the overall accuracy of the classifier on the test set'''
    
    best_class = np.argmax(classification, axis=1)
    test_docs,labels = get_all_docs(test_client, subset_field)
    if project_type == 'combined': 
        _,_,_,labels = split_train_test(test_docs, labels)
    gold = np.array([list(classifiers['simple'].classes_).index(label) for label in labels])
    accuracy = (gold == best_class).sum() / len(gold)
    return accuracy

def main(args):
    '''Collect required arguments if not supplied'''
    
    if not args.account_id:
        args.account_id = input('Enter the account id: ')
    if not args.training_project_id:
        args.training_project_id = input('Enter the id of the training project: ')
    if not args.testing_project_id:
        args.testing_project_id = input('Enter the id of the testing project: ')
    if not args.subset_field:
        args.subset_field = input('Subset field holding the label("Category label"): ')
    
    client = LuminosoClient.connect()
    train_client = client.change_path('/projects/{}/{}'.format(args.account_id,args.training_project_id))
    test_client = client.change_path('/projects/{}/{}'.format(args.account_id,args.testing_project_id))
    
    '''For demo purposes, a single project can be used ("combined"), 
        for POC purposes, projects should be split into training & test.'''
    if args.testing_project_id==args.training_project_id:
        project_type = 'combined'
    else:
        project_type = 'separate'
    
    # Allows for live demo-ing in Python notebook
    if args.live:
        print('TRAINING CLASSIFIER...')
        classifiers, vectorizers = train_classifier(train_client, args.subset_field, project_type)
        print('CLASSIFIER TRAINED. ENTER EXAMPLE TEXT BELOW OR "exit" TO EXIT.\n\n')
        while True:
            new_text = input('Enter text to be classified: ')
            if new_text == 'exit':
                break
            else:
                print('The predicted value is: "{0}".\n The model is {1:.2%} confident.\n'.format(*return_label(new_text, classifiers, vectorizers, train_client)))
    else:
        classifiers, vectorizers = train_classifier(train_client, args.subset_field, project_type)
        classification = test_classifier(train_client, test_client, classifiers, vectorizers, args.subset_field, project_type, args.save_results)
        print('Accuracy:{}%'.format(score_results(test_client, classifiers, classification, args.subset_field, project_type)))
    
if __name__ == '__main__':      
    parser = argparse.ArgumentParser(
        description='Create a classification model based on an existing project using subsets as labels.'
    )
    parser.add_argument('-a','--account_id', help="The ID of the account that owns the project, such as 'demo'")
    parser.add_argument('-tr','--training_project_id', help="The ID of the project that contains the training data")
    parser.add_argument('-te','--testing_project_id', help="The ID of the project that contains the training data")
    parser.add_argument('-f', '--subset_field', help="The name of the subset field being classified against, such as 'label:'")
    parser.add_argument('-l', '--live', help="The name of the predict field being classified against, such as 'label'", default=False, action='store_true')
    parser.add_argument('-s', '--save_results', help="Save the results of the test set to a CSV file named results.csv", default=False, action='store_true')
    args = parser.parse_args()
    main(args)