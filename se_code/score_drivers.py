from luminoso_api import LuminosoClient
from pack64 import unpack64
from sklearn.linear_model import Ridge
import numpy as np
import csv, time, sys, argparse, getpass, json

def get_as(vector1, vector2):
    '''
    Calculate the association score between two vectors
    :param vector1: First vector
    :param vector2: Second vector
    :return: Cosine similarity of two vectors
    '''
    return np.dot([float(v) for v in unpack64(vector1)], [float(v) for v in unpack64(vector2)])

def is_number(s):
    '''
    Detect whether a string is a number
    :param s: string to be tested
    :return: True/False
    '''
    try:
        float(s)
        return True
    except ValueError:
        return False
        
def add_score_drivers_to_project(client, docs, drivers):
    '''
    Create add data to 'predict' field to support creation of ScoreDrivers if none existed
    :param client: LuminosoClient object pointed to project path
    :param docs: List of document dictionaries
    :param drivers: List of subsets (string) that contain numerics (could be score drivers)
    :return: None
    '''
   
    mod_docs = []
    for doc in docs:
        predict = {}
        for subset_to_score in drivers:
            if subset_to_score in [a.split(':')[0] for a in doc['subsets']]:
                predict.update({subset_to_score: float([a for a in doc['subsets'] 
                         if subset_to_score.strip().lower() == a.split(':')[0].strip().lower()][0].split(':')[-1])})
        mod_docs.append({'_id': doc['_id'],
                         'predict': predict})
    client.put_data('docs', json.dumps(mod_docs), content_type='application/json')
    client.post('docs/recalculate')

    wait_for_jobs(client, 'recalculation')
    print('Done recalculating. Training...')
    client.post('prediction/train')
    wait_for_jobs(client, 'driver training')
    print('Done training.')
    
def create_drivers_table(client, drivers, topic_drive):
    '''
    Create tabulation of ScoreDrivers output, complete with doc counts, example docs, scores and driver clusters
    :param client: LuminosoClient object pointed to project path
    :param drivers: List of drivers (string)
    :param topic_drive: Whether or not to include topics as drivers (bool)
    :return: List of drivers with scores, example docs, clusters and type
    '''

    driver_table = []
    for subset in drivers:
        if topic_drive:
            topic_drivers = client.put('prediction/drivers', predictor_name=subset)
            for driver in topic_drivers:
                row = {}
                row['driver'] = driver['text']
                row['type'] = 'user_defined'
                row['subset'] = subset
                row['impact'] = driver['regressor_dot']
                row['score'] = driver['driver_score']
                # ADDED RELATED TERMS
                related_terms = driver['terms']
                char_length_count = 0
                terms_to_get = []
                list_terms = []
                for k, r in enumerate(related_terms):
                    char_length_count += len(r)
                    terms_to_get.append(r)
                    if char_length_count > 250 or (k == len(related_terms) - 1):
                        list_terms.extend(client.get('terms', terms=terms_to_get))
                        char_length_count = 0
                        terms_to_get = []
                doc_count_terms_list = [related_terms[0]]
                related_text = []
                for term in list_terms:
                    related_text.append(term['text'])
                row['related_terms'] = related_text
                doc_count = client.get('terms/doc_counts', terms=doc_count_terms_list, use_json=True)
                count_sum = 0
                for doc_dict in doc_count:
                    count_sum += (doc_dict['num_exact_matches'] + doc_dict['num_related_matches'])
                row['doc_count'] = count_sum

                # Use the driver term to find related documents
                search_docs = client.get('docs/search', terms=driver['terms'], limit=500, exact_only=True)

                # Sort documents based on their association with the coefficient vector
                for doc in search_docs['search_results']:
                    document = doc[0]['document']
                    document['driver_as'] = get_as(driver['vector'],document['vector'])

                docs = sorted(search_docs['search_results'], key=lambda k: k[0]['document']['driver_as']) 
                row['example_doc'] = ''
                row['example_doc2'] = ''
                row['example_doc3'] = ''
                if len(docs) >= 1:
                    row['example_doc'] = docs[0][0]['document']['text']
                if len(docs) >= 2:
                    row['example_doc2'] = docs[1][0]['document']['text']
                if len(docs) >= 3:
                    row['example_doc3'] = docs[2][0]['document']['text']
                driver_table.append(row)
        try:
            score_drivers = client.get('prediction/drivers', predictor_name=subset)
            for driver in score_drivers['negative']:
                row = {}
                row['driver'] = driver['text']
                row['type'] = 'auto_found'
                row['subset'] = subset
                row['impact'] = driver['regressor_dot']
                row['score'] = driver['driver_score']
                # ADDED RELATED TERMS
                related_terms = driver['similar_terms']
                char_length_count = 0
                terms_to_get = []
                list_terms = []
                for k, r in enumerate(related_terms):
                    char_length_count += len(r)
                    terms_to_get.append(r)
                    if char_length_count > 250 or (k == len(related_terms) - 1):
                        list_terms.extend(client.get('terms', terms=terms_to_get))
                        char_length_count = 0
                        terms_to_get = []
                #list_terms = client.get('terms', terms=related_terms)
                doc_count_terms_list = [related_terms[0]]
                related_text = []
                for term in list_terms:
                    related_text.append(term['text'])
                row['related_terms'] = related_text
                doc_count = client.get('terms/doc_counts', terms=doc_count_terms_list, use_json=True)
                count_sum = 0
                for doc_dict in doc_count:
                    count_sum += (doc_dict['num_exact_matches'] + doc_dict['num_related_matches'])
                row['doc_count'] = count_sum


                    # Use the driver term to find related documents
                search_docs = client.get('docs/search', terms=[driver['term']], limit=500, exact_only=True)

                    # Sort documents based on their association with the coefficient vector
                for doc in search_docs['search_results']:
                    document = doc[0]['document']
                    document['driver_as'] = get_as(score_drivers['coefficient_vector'],document['vector'])

                docs = sorted(search_docs['search_results'], key=lambda k: k[0]['document']['driver_as']) 
                row['example_doc'] = ''
                row['example_doc2'] = ''
                row['example_doc3'] = ''
                if len(docs) >= 1:
                    row['example_doc'] = docs[0][0]['document']['text']
                if len(docs) >= 2:
                    row['example_doc2'] = docs[1][0]['document']['text']
                if len(docs) >= 3:
                    row['example_doc3'] = docs[2][0]['document']['text']
                driver_table.append(row)
            for driver in score_drivers['positive']:
                row = {}
                row['driver'] = driver['text']
                row['type'] = 'auto_found'
                row['subset'] = subset
                row['impact'] = driver['regressor_dot']
                row['score'] = driver['driver_score']
                related_terms = driver['similar_terms']
                char_length_count = 0
                terms_to_get = []
                list_terms = []
                for k, r in enumerate(related_terms):
                    char_length_count += len(r)
                    terms_to_get.append(r)
                    if char_length_count > 250 or (k == len(related_terms) - 1):
                        list_terms.extend(client.get('terms', terms=terms_to_get))
                        char_length_count = 0
                        terms_to_get = []
                doc_count_terms_list = [related_terms[0]]
                related_text = []
                for term in list_terms:
                    related_text.append(term['text'])
                row['related_terms'] = related_text
                doc_count = client.get('terms/doc_counts', terms=doc_count_terms_list, use_json=True)
                count_sum = 0
                for doc_dict in doc_count:
                    count_sum += (doc_dict['num_exact_matches'] + doc_dict['num_related_matches'])
                row['doc_count'] = count_sum

                    # Use the driver term to find related documents
                search_docs = client.get('docs/search', terms=[driver['term']], limit=500, exact_only=True)

                    # Sort documents based on their association with the coefficient vector
                for doc in search_docs['search_results']:
                    document = doc[0]['document']
                    document['driver_as'] = get_as(score_drivers['coefficient_vector'],document['vector'])

                docs = sorted(search_docs['search_results'], key=lambda k: -k[0]['document']['driver_as'])
                row['example_doc'] = ''
                row['example_doc2'] = ''
                row['example_doc3'] = ''
                if len(docs) >= 1:
                    row['example_doc'] = docs[0][0]['document']['text']
                if len(docs) >= 2:
                    row['example_doc2'] = docs[1][0]['document']['text']
                if len(docs) >= 3:
                    row['example_doc3'] = docs[2][0]['document']['text']
                driver_table.append(row)
        except:
            print('WARNING: %s is not a trained regressor, skipping...' % subset)
            continue
    
    return driver_table
    
def get_all_docs(client):
    docs = []
    while True:
        new_docs = client.get('docs', limit=25000, offset=len(docs))
        if new_docs:
            docs.extend(new_docs)
        else:
            return docs

def write_table_to_csv(table, filename):
    '''
    Function for writing lists of dictionaries to a CSV file
    :param table: List of dictionaries to be written
    :param filename: Filename to be written to (string)
    :return: None
    '''

    print('Writing to file {}.'.format(filename))
    if len(table) == 0:
        print('Warning: No data to write to {}.'.format(filename))
        return
    try:
        with open(filename, 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
    except UnicodeEncodeError as e:
        print('WARNING: Unicode Decode Error occurred, attempting to handle. Error was: %s' % e)
        write_table = [{k: v for k, v in t.items()} for t in table]
        with open(filename, 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=write_table[0].keys())
            writer.writeheader()
            writer.writerows(write_table)
        print('Unicode Decode Error was handled')

def wait_for_jobs(client, text):
    '''
    Repeatedly test for project recalculation, display text when complete
    :param client: LuminosoClient object pointed to project path
    :param text: String to print when complete
    :return: None
    '''

    time_waiting = 0
    while len(client.get()['running_jobs']) != 0:
        sys.stderr.write('\r\tWaiting for {} ({}sec)'.format(text, time_waiting))
        time.sleep(30)
        time_waiting += 30
    
def main():
    parser = argparse.ArgumentParser(
        description='Export Subset Key Terms and write to a file'
    )
    parser.add_argument('project_url', help="The complete URL of the Analytics project")
    parser.add_argument('--rebuild', default=False, action='store_true', help="If set, will always rebuild drivers based on numeric subsets, regardless of existing ones")
    parser.add_argument('--topic_drivers', default=False, action='store_true', help="If set, will calculate drivers based on user-defined topics as well")
    args = parser.parse_args()
    
    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v4'
    account_id = project_url.split('/')[-2].strip()
    project_id = project_url.split('/')[-1].strip()
    
    count = 0
    while count < 3:
        username = input('Username: ')
        password = getpass.getpass()
        try:
            client = LuminosoClient.connect(url='%s/projects/%s/%s' % (api_url.strip('/'), account_id, project_id), username=username, password=password)
            break
        except:
            print('Incorrect credentials, please re-enter username and password')
            count += 1
            continue
    if count >= 3:
        print('Username and password incorrect')
        
    print('Getting Drivers...')
    docs = get_all_docs(client)
    subsets = client.get('subsets/stats')
    drivers = list(set([key for d in docs for key in d['predict'].keys()]))
    exist_flag = True

    # See if any score drivers are present, if not, create some from subsets
    if not any(drivers):
        exist_flag = False
        drivers = []
        subset_headings = list(set([s['subset'].partition(':')[0] for s in subsets]))
        for subset in subset_headings:
            subset_values = [s['subset'].partition(':')[2] for s in subsets
                             if s['subset'].partition(':')[0] == subset]
            if all([is_number(v) for v in subset_values]):
                drivers.append(subset)
    
    if args.rebuild or not exist_flag:
        add_score_drivers_to_project(client, docs, drivers)
        
    driver_table = create_drivers_table(client, drivers, args.topic_drivers)
    write_table_to_csv(driver_table, 'drivers_table.csv')
    
    
if __name__ == '__main__':
    main()
