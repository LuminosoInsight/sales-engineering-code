from luminoso_api import LuminosoClient
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from pack64 import pack64, unpack64
import numpy as np
import argparse
import math
import csv
import json
    
def sigmoid(x):
    '''
    Map distance to hyperplane to a range 0-1. Approximation of "likelihood" that the prediction is correct.
    x being a list of lists, aka list of the output the from decision_function for each sample 
    '''
    
    return [[1 / (1 + math.exp(-y)) for y in z] for z in x]

def combine_decision_functions(cls_dfuncs, classes, weights=None):
    '''
    Combine outputs from multiple classifiers. Apply weights. Take mean of result across classifiers.
    The weighting ability of this function has yet to show meaningful improvements in accuracy... remove?
    '''
 
    if classes == 2:
        cls_dfuncs = [[(-b,b) for b in a] for a in cls_dfuncs]
        
    if weights:
        cls_funcs = [c*w for c,w in zip(cls_dfuncs,weights)]
    
    classification = np.dstack(cls_dfuncs)
    
    classification = np.mean(classification, axis=2)  
    
    return sigmoid(classification)
  
def merge_two_dicts(x, y):
    '''
    Given two dicts, merge them into a new dict as a shallow copy.
    '''
    
    z = x.copy()
    z.update(y)
    return z

def sklearn_text(termlist, lang='en'):
    '''
    Convert a list of Luminoso terms, possibly multi-word terms, into text that
    the tokenizer we get from `make_term_vectorizer` below will tokenize into
    those terms.

    Yes, the tokenizer will basically be undoing what this function does, but it
    means we also get the benefit of sklearn's TF-IDF.
    '''
    langtag = '|' + lang
    fixed_terms = [
        term.replace(langtag, '').replace(' ', '_')
        for term, _tag, _span in termlist
        if '\N{PILCROW SIGN}' not in term
    ]
    return ' '.join(fixed_terms)

def get_all_docs(client, subset_field, batch_size=20000):
    '''
    Pull all docs from project, using a particular subset as the LABEL
    '''
    
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
    '''
    Split documents & labels into a single dict each for both testing and training
    '''
    
    labels_without_enough_samples = [label for label in set(labels) if labels.count(label)==1]
    if labels_without_enough_samples:
        indexes = [i for i,label in enumerate(labels) if label not in labels_without_enough_samples]
        labels = [labels[i] for i in indexes]
        docs = [docs[i] for i in indexes]
    return train_test_split(np.array(docs), np.array(labels), test_size=split, random_state=32, stratify=np.array(labels))
        
def make_term_vectorizer():
    '''
    Return a sklearn vectorizer whose tokenizer only splits on whitespace.
    This is for text that we have already tokenized, in the Luminoso way, and
    then stuck together with whitespace.
    '''
    
    # Consider using min_df?
    return TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=None, token_pattern=r'\S+')

def make_simple_vectorizer():
    '''
    Return a sklearn vectorizer that does sklearn's usual thing with arbitrary
    English text.
    '''
    
    # Consider using min_df?
    return TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

def train_classifier(client, train_docs, train_labels):
    '''
    Train a classifier.

    Input: a list of training documents, and a list of labels.
    Output: a pair of (classifiers, vectorizers).

    The returned items represent three different sklearn classifiers and their
    corresponding vectorizers. These should be passed on to the `test_classifier`
    function.
    '''
    
    assert len(train_docs) > 0
    
    term_vectorizer = make_term_vectorizer()
    simple_vectorizer = make_simple_vectorizer()
    vectorizers = {'simple': simple_vectorizer, 'term': term_vectorizer}
    
    classifiers = {
        style: LinearSVC(
            C=1.0, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr',
            penalty='l1', random_state=None, tol=0.0001, verbose=0
        )
        for style in ('simple', 'term', 'vector')
    }
    
    simple_vecs = vectorizers['simple'].fit_transform([doc['text'] for doc in train_docs])
    term_vecs = vectorizers['term'].fit_transform([sklearn_text(doc['terms']) for doc in train_docs])
    luminoso_vecs = normalize([unpack64(doc['vector']) for doc in train_docs])
    
    classifiers['simple'].fit(simple_vecs, train_labels)
    classifiers['term'].fit(term_vecs, train_labels)
    classifiers['vector'].fit(luminoso_vecs, train_labels)
    
    return (classifiers, vectorizers)

def binary_rating_labeler(rating):
    '''
    An example that produces labels from Amazon book reviews.

    The classes it produces are 'pos' or 'neg' depending on whether the
    document has a 'rating' of more or less than 3. It returns None for a rating
    of exactly 3, saying to skip that document.
    '''
    
    if rating == 3:
        return None
    return ('pos' if rating > 3 else 'neg')

def get_test_docs_from_file(filename, label_func=None):
    '''
    Test data consists of dictionaries with 'text' and 'label' values. It doesn't
    need other fields. This means it can come from outside of a Luminoso project
    if necessary.
    
    label_func specifies a transformation function applied to inbound labels
    '''
    
    all_docs = []
    all_labels = []
    with open(filename) as infile:
        n_docs = 0
        for i, line in enumerate(infile):
            doc = json.loads(line.rstrip())

            if label_func:
                label = label_func(doc['label'])
            else:
                label = doc['label']
            
            if label is None:
                continue

            all_docs.append({
                'text': doc['text']})
            
            all_labels.append(label)
            
    return all_docs, all_labels

def test_classifier(train_client, test_docs, test_labels, classifiers, vectorizers, save_results=False):
    '''
    Inputs:

    * `train_client`: a LuminosoClient pointing to the root of a project, which will
      be used to vectorize the test documents.
    * `test_docs`: test documents, which must have at least a `text` item.
    * `test_labels`: test labels, which must be in same order as test_docs.
    * `classifiers` and `vectorizers` are the result of running
      `train_classifier` on training data.
    * 'save_results' flag to indicate whether results should be saved to a CSV for exploration/visualization

    Returns a list of classes assigned to the documents in order, and the
    decision matrix, whose dimensions are (n_docs, n_classes).
    '''
        
    test_docs = train_client.upload('docs/vectors', test_docs)
    simple_vecs = vectorizers['simple'].transform([doc['text'] for doc in test_docs])
    term_vecs = vectorizers['term'].transform([sklearn_text(doc['terms']) for doc in test_docs])
    luminoso_vecs = normalize([unpack64(doc['vector']) for doc in test_docs])

    classification = combine_decision_functions([
        classifiers['simple'].decision_function(simple_vecs),
        classifiers['term'].decision_function(term_vecs),
        classifiers['vector'].decision_function(luminoso_vecs)
    ],
            len(classifiers['simple'].classes_))
        
    if save_results:
        results_dict = [merge_two_dicts({'text': z[0]['text'],'truth': z[1]},dict(zip(list(classifiers['simple'].classes_),z[2]))) for z in zip(test_docs,test_labels,classification)]
        writer = csv.DictWriter(open('results.csv','w',encoding='utf-8'),['text','truth']+list(classifiers['simple'].classes_))
        writer.writeheader()
        writer.writerows(results_dict)
        
    return classification

def return_label(new_text, classifiers, vectorizers, train_client):
    '''
    Return label function for operating in a live demo
    Returns best class and "confidence score"
    '''
    
    test_doc = train_client.upload('docs/vectors', [{'text':new_text}])[0]
    simple_vecs = vectorizers['simple'].transform([test_doc['text']])
    term_vecs = vectorizers['term'].transform([sklearn_text(test_doc['terms'])])
    luminoso_vecs = normalize([unpack64(test_doc['vector'])])

    classification = combine_decision_functions([
        classifiers['simple'].decision_function(simple_vecs),
        classifiers['term'].decision_function(term_vecs),
        classifiers['vector'].decision_function(luminoso_vecs)
    ],
            len(classifiers['simple'].classes_))

    best_class = np.argmax(classification, axis=1)[0]
    return classifiers['simple'].classes_[best_class],classification[0][best_class]

def score_results(test_labels, classifiers, classification):
    '''
    Return the overall accuracy of the classifier on the test set
    '''
    
    best_class = np.argmax(classification, axis=1)
    gold = np.array([list(classifiers['simple'].classes_).index(label) for label in test_labels]) #Fails if test labels don't match training labels
    accuracy = (gold == best_class).sum() / len(gold)
    return accuracy

def main(args):
    '''
    Collect required arguments if not supplied
    '''
    
    if not args.account_id:
        args.account_id = input('Enter the account id: ')
    if not args.training_project_id:
        args.training_project_id = input('Enter the id of the training project: ')
    if not args.testing_data:
        args.testing_data = input('Enter the id of the testing project: ')
    if not args.subset_field:
        args.subset_field = input('Subset field holding the label("Category label"): ')
    
    client = LuminosoClient.connect(url=args.url,username=args.username)
        
    train_client = client.change_path('/projects/{}/{}'.format(args.account_id,args.training_project_id))
    
    '''
    For demo purposes, a single project can be used for both training/testing, 
    for POC purposes, projects should be split into training & test.
    '''
    if args.test_file:
        test_docs, test_labels = get_test_docs_from_file(args.testing_data)
        train_docs, train_labels = get_all_docs(train_client, args.subset_field)
    else:
        if args.testing_data==args.training_project_id:
            docs,labels = get_all_docs(train_client, args.subset_field)
            train_docs,test_docs,train_labels,test_labels = split_train_test(docs,labels)
        else:
            test_client = client.change_path('/projects/{}/{}'.format(args.account_id,args.testing_data))
            train_docs, train_labels = get_all_docs(train_client, args.subset_field)
            test_docs, test_labels = get_all_docs(test_client, args.subset_field)
    
    # Allows for live demo-ing in Python notebook
    if args.live:
        print('Training classifier...')
        classifiers, vectorizers = train_classifier(train_client, train_docs, train_labels)
        print('Classifier trained. Enter example text below or "exit" to exit.\n\n')
        while True:
            new_text = input('Enter text to be classified: ')
            if new_text == 'exit':
                break
            else:
                print('The predicted value is: "{0}".\n The model is {1:.2%} confident.\n'.format(*return_label(new_text, classifiers, vectorizers, train_client)))
    else:
        classifiers, vectorizers = train_classifier(train_client, train_docs, train_labels)
        classification = test_classifier(train_client, test_docs, test_labels, classifiers, vectorizers, args.save_results)
        print('Accuracy:{}%'.format(score_results(test_labels, classifiers, classification)))
    
if __name__ == '__main__':      
    '''
    BENCHMARK PROJECTS
    USAA: (-a a53y655v -tr 54hdb -td 9b2fw -f "Label: ") Accuracy:0.6997713165632147%
    Pandora: (-a h82y756m -tr vnfzx -td vnfzx -f "Category Tag: ") Accuracy:0.8111111111111111%
    Fidelity: (-a a53y655v -tr sv5pn -td sv5pn -f "CED: ") Accuracy:0.8308142940831869%
    Fidelity: (-a a53y655v -tr sv5pn -td sv5pn -f "COSMO_SEMANTIC_TAG: ") Accuracy:0.80199179847686%
    SuperCell: (-a a53y655v -tr 6bsv2 -td 6bsv2 -f "Type: ") Accuracy:0.833%
    Subaru: (-a a53y655v -tr fpdxb -td fpdxb -f "Site: ") 
    '''
    
    parser = argparse.ArgumentParser(
        description='Create a classification model based on an existing project using subsets as labels.'
    )
    parser.add_argument('-u','--username', help='Username (email) of Luminoso account')
    parser.add_argument('-url','--url', help='URL of Luminoso API endpoint (https://eu-analytics.luminoso.com/api/v4)')
    parser.add_argument('-a','--account_id', help="The ID of the account that owns the project, such as 'demo'")
    parser.add_argument('-tr','--training_project_id', help="The ID of the project that contains the training data")
    parser.add_argument('-td','--testing_data', help="The ID of the project, or filename that contains the testing data")
    parser.add_argument('-file', '--test_file', help="CSV file with testing data: (text,label) columns", default=False, action='store_true')
    parser.add_argument('-f', '--subset_field', help="The name of the subset field being classified against, such as 'label:'")
    parser.add_argument('-l', '--live', help="The name of the predict field being classified against, such as 'label'", default=False, action='store_true')
    parser.add_argument('-s', '--save_results', help="Save the results of the test set to a CSV file named results.csv", default=False, action='store_true')
    args = parser.parse_args()
    main(args)
