import csv

from sklearn.model_selection import train_test_split
import numpy as np

encoding_character = '\ufeff'

def get_docs_labels(docs, subset_field, batch_size=20000):
    '''
    Pull all docs from project, using a particular subset as the LABEL
    '''
    if ':' in subset_field:
        subset_field = subset_field.lower().strip(': ')
        
    new_docs = []
    labels = []
    for doc in docs:
        new_docs.append(doc)
        labels.append(doc[subset_field])

    return docs, labels
        
def split_train_test(docs, labels, split=0.3):
    '''
    Split documents & labels into a single dict for both testing and training
    '''

    labels_without_enough_samples = [label for label in set(labels)
                                     if labels.count(label) == 1]

    if labels_without_enough_samples:
        indexes = [i
                   for i, label in enumerate(labels)
                   if label not in labels_without_enough_samples]

        labels = [labels[i] for i in indexes]
        docs = [docs[i] for i in indexes]
    return train_test_split(np.array(docs), np.array(labels), test_size=split, random_state=32, stratify=np.array(labels))

def main():
    parser = argparse.ArgumentParser(
        description='Split a given data file into a stratified training and testing set.'
    )
    parser.add_argument('read_file', help="The name of the CSV containing documents. Should have a 'text' and 'label' column")
    parser.add_argument('--train_file', default=None, help="name of the file to write the training set to")
    parser.add_argument('--test_file', default=None, help="name of the file to write the testing set to")
    parser.add_argument('--text_index', default= "text", help="name of the column containing the text")
    parser.add_argument('--label_index', default= "label", help="name of the column containing the label")
    parser.add_argument('--split', type=float, default=.3, help="The percentage of documents to reserve for testing")
    
    args = parser.parse_args()
    
    with open(read_file, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        table = [row for row in reader]
    keys = list(table[0].keys())
    for k in keys:
        if args.text_index in k and encoding_character in k:
            args.text_index = encoding_character + args.text_index
        if args.label_index in k and encoding_character in k:
            args.label_index = encoding_character + args.label_index
    table = [{'text': t.get(args.text_index), 'label': t.get(args.label_index)} for t in table]
    
    docs = []
    unlabeled_docs = []
    for t in table:
        if not t.get('label') or t['label'].strip() == '':
            unlabeled_docs.append(t)
        else:
            docs.append(t)
            
    docs, labels = get_docs_labels(docs, 'label')
    train_docs, test_docs, train_labels, test_labels = split_train_test(docs, labels, args.split)
    train_docs = list(train_docs)
    train_docs.extend(unlabeled_docs)
    
    if args.train_file:
        train_write_file = args.train_file
    else:
        train_write_file = read_file.split('.')[0] + '_training.csv'
    if args.test_file:
        test_write_file = args.test_file
    else:
        test_write_file = read_file.split('.')[0] + '_testing.csv'
    
    with open(train_write_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, ['text', 'label'])
        writer.writeheader()
        writer.writerows(train_docs)
    
    with open(test_write_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, ['text', 'label'])
        writer.writeheader()
        writer.writerows(test_docs)

    
if __name__ == '__main__':
    main()
