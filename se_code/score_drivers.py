from luminoso_api import V5LuminosoClient as LuminosoClient
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
    
def get_driver_fields(client):
    '''
    Get all numeric metadata fields from the project in order to run drivers against
    :param client: LuminosoClient object pointed to project path
    :return: List of fields that contain drivers
    '''
    metadata = client.get('metadata')
    driver_fields = [m['name'] for m in metadata['result'] if m['type'] == 'number']
    return driver_fields
    
    
def create_drivers_table(client, driver_fields, topic_drive):
    '''
    Create tabulation of ScoreDrivers output, complete with doc counts, example docs, scores and driver clusters
    :param client: LuminosoClient object pointed to project path
    :param driver_fields: List of driver fields (string list)
    :param topic_drive: Whether or not to include topics as drivers (bool)
    :return: List of drivers with scores, example docs, clusters and type
    '''
    driver_table = []
    for field in driver_fields:
        if topic_drive:
            score_drivers = client.get('concepts/score_drivers', score_field=field,
                                       concept_selector={'type': 'saved'})
            for driver in score_drivers:
                row = {}
                row['driver'] = driver['name']
                row['type'] = 'user_defined'
                row['subset'] = field
                row['impact'] = driver['impact']
                row['related_terms'] = driver['texts']
                row['doc_count'] = driver['exact_match_count']

                # Use the driver term to find related documents
                search_docs = client.get('docs', search={'texts': driver['texts']}, limit=500, exact_only=True)

                # Sort documents based on their association with the coefficient vector
                for doc in search_docs['result']:
                    document['driver_as'] = get_as(driver['vector'],document['vector'])

                docs = sorted(search_docs['result'], key=lambda k: k['driver_as']) 
                row['example_doc'] = ''
                row['example_doc2'] = ''
                row['example_doc3'] = ''
                if len(docs) >= 1:
                    row['example_doc'] = docs[0]['text']
                if len(docs) >= 2:
                    row['example_doc2'] = docs[1]['text']
                if len(docs) >= 3:
                    row['example_doc3'] = docs[2]['text']
                driver_table.append(row)
        score_drivers = client.get('concepts/score_drivers', score_field=field, limit=100)
        score_drivers = [d for d in score_drivers if d['importance'] >= .4]
        for driver in score_drivers:
            row = {}
            row['driver'] = driver['name']
            row['type'] = 'auto_found'
            row['subset'] = field
            row['impact'] = driver['impact']
            row['related_terms'] = driver['texts']
            row['doc_count'] = driver['exact_match_count']

            # Use the driver term to find related documents
            search_docs = client.get('docs', search={'texts': driver['texts']}, limit=500, exact_only=True)

            # Sort documents based on their association with the coefficient vector
            for doc in search_docs['result']:
                doc['driver_as'] = get_as(driver['vector'],doc['vector'])

            docs = sorted(search_docs['result'], key=lambda k: k['driver_as']) 
            row['example_doc'] = ''
            row['example_doc2'] = ''
            row['example_doc3'] = ''
            if len(docs) >= 1:
                row['example_doc'] = docs[0]['text']
            if len(docs) >= 2:
                row['example_doc2'] = docs[1]['text']
            if len(docs) >= 3:
                row['example_doc3'] = docs[2]['text']
            driver_table.append(row)
    return driver_table
    
    
def get_all_docs(client):
    docs = []
    while True:
        new_docs = client.get('docs', limit=25000, offset=len(docs))['result']
        if new_docs:
            docs.extend(new_docs)
        else:
            return docs

def write_table_to_csv(table, filename, encoding='utf-8'):
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
        with open(filename, 'w', encoding=encoding, newline='') as file:
            writer = csv.DictWriter(file, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
    except UnicodeEncodeError as e:
        print('WARNING: Unicode Decode Error occurred, attempting to handle. Error was: %s' % e)
        write_table = [{k: v for k, v in t.items()} for t in table]
        with open(filename, 'w', encoding=encoding, newline='') as file:
            writer = csv.DictWriter(file, fieldnames=write_table[0].keys())
            writer.writeheader()
            writer.writerows(write_table)
        print('Unicode Decode Error was handled')
    
def main():
    parser = argparse.ArgumentParser(
        description='Export Subset Key Terms and write to a file'
    )
    parser.add_argument('project_url', help="The complete URL of the Analytics project")
    parser.add_argument('-t', '--token', default=None, help="Authentication Token for Daylight")
    parser.add_argument('--topic_drivers', default=False, action='store_true', help="If set, will calculate drivers based on user-defined topics as well")
    parser.add_argument('--encoding', default='utf-8', help="Encoding type of the files to write to")
    args = parser.parse_args()
    
    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v5'
    account_id = project_url.split('/')[-2].strip()
    project_id = project_url.split('/')[-1].strip()
    
    if args.token:
        client = LuminosoClient.connect(url='%s/projects/%s' % (api_url.strip('/'), project_id), token=args.token)
    else:
        client = LuminosoClient.connect(url='%s/projects/%s' % (api_url.strip('/'), project_id))
        
    print('Getting Drivers...')
    docs = get_all_docs(client)
    
    driver_fields = get_driver_fields(client)
    driver_table = create_drivers_table(client, drivers, args.topic_drivers)
    write_table_to_csv(driver_table, 'drivers_table.csv', encoding=args.encoding)
    
    
if __name__ == '__main__':
    main()
