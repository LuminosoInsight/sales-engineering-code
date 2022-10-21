import argparse,csv,os
from sklearn.model_selection import train_test_split
import numpy as np


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

def create_train_test(read_file, train_write_file, test_write_file, split, encoding, text_index, label_index):

    encoding_character = '\ufeff'

    # read the raw data
    with open(read_file,encoding=encoding) as f:
        reader = csv.DictReader(f)
        table = [row for row in reader]

    # filter encoding characters
    keys = list(table[0].keys())
    change = False
    for k in keys:
        if text_index in k and encoding_character in k:
            text_index = encoding_character + text_index
            text_index = text_index
            change = True
        if label_index in k and encoding_character in k:
            label_index = encoding_character + label_index
            label_index = label_index
            change = True

    # filter unlabeled docs
    docs = []
    unlabeled_docs = []
    for t in table:
        if not t.get(label_index) or t[label_index].strip() == '':
            unlabeled_docs.append(t)
        else:
            docs.append(t)

    # build the train and test sets
    docs, labels = get_docs_labels(docs, label_index)
    train_docs, test_docs, train_labels, test_labels = split_train_test(docs, labels, split)
    train_docs = list(train_docs)
    train_docs.extend(unlabeled_docs)

    # output the data
    with open(train_write_file, 'w', encoding=encoding) as f:
        writer = csv.DictWriter(f, train_docs[0].keys(),lineterminator='\n')
        writer.writeheader()
        writer.writerows(train_docs)
        
    with open(test_write_file, 'w', encoding=encoding) as f:
        writer = csv.DictWriter(f, test_docs[0].keys(),lineterminator='\n')
        writer.writeheader()
        writer.writerows(test_docs)

def main():

    parser = argparse.ArgumentParser(
        description='Export Subset Key Terms and write to a file'
    )

    parser.add_argument('--input', default='input.csv', help="Filename to use as input. Default=input.csv")
    parser.add_argument('--train', default='output_train.csv', help="Filename to use as training set output. Default: output_train.csv")
    parser.add_argument('--test', default='output_test.csv', help="Filename to use as test set output. Default: output_test.csv")
    parser.add_argument('--encoding', default='utf-8-sig', help="Encoding type of the files to write to. Default: utf-8-sig")
    parser.add_argument('--split_size', default='0.3', help="Size of the test set. 0.3 default")
    parser.add_argument('--text_index', default='text', help="Name of the column that has the text field")
    parser.add_argument('--label_index', default='label', help="Name of the column that has the label field")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("Invalid input filename: {}".format(args.input))
        parser.print_help()
        return

    encoding = args.encoding

    read_file = args.input
    train_write_file = args.train
    test_write_file =  args.test 

    text_index = args.text_index
    label_index = args.label_index

    # split is the size of the test set
    split = float(args.split_size)

    create_train_test(read_file, train_write_file, test_write_file, split, encoding, text_index, label_index)

if __name__ == '__main__':
    main()
