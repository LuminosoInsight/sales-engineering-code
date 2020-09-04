from luminoso_api import V5LuminosoClient as LuminosoClient
from pack64 import unpack64
from sklearn.linear_model import Ridge
import numpy as np
import csv, time, sys, argparse, getpass, json
import urllib

from datetime import datetime
from datetime import timedelta
import concurrent.futures
import pandas as pd

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
    Get all numeric or score metadata fields from the project in order to run drivers against
    :param client: LuminosoClient object pointed to project path
    :return: List of fields that contain drivers
    '''
    metadata = client.get('metadata')
    driver_fields = [m['name'] for m in metadata['result'] if m['type'] == 'number' or m['type'] == 'score']
    return driver_fields

def get_first_date_field(client,warn_too_many_dates=False):
    '''
    Get the first date field
    :param client: LuminosoClient object pointed to project path
    :return: dictionary with the date field info
    '''
    metadata = client.get('metadata')

    if len([df['name'] for df in metadata['result'] if df['type']=='date'])>1:
        print("WARNING: multiple date fields. Using first date field found.")

    for df in metadata['result']:
        if (df['type']=='date'):
            return df
    return None

def get_date_field_by_name(client,date_field_name):
    '''
    Get the date field by name
    :param client: LuminosoClient object pointed to project path
    :return: dictionary with the date field info
    '''
    metadata = client.get('metadata')
    for df in metadata['result']:
        if (df['name']==date_field_name):
            return df
    return None

def find_best_interval(client,docs,date_field_name,num_intervals):
    docs_by_id = [{d['doc_id']:d} for d in docs]
    docs_by_date = []
    for i,d in enumerate(docs):
        for m in d['metadata']:
            if (m['name']==date_field_name):
                docs_by_date.append({'date':datetime.strptime(m['value'], '%Y-%m-%dT%H:%M:%S.%fZ'),'doc_id':d['doc_id'],'i':1})
                break

    df = pd.DataFrame(docs_by_date)
    df.set_index(['date'])
    pd.to_datetime(df.date,unit='s')
    
    interval_types = ['M','W','D']
    df = pd.DataFrame(docs_by_date)
    #df['date'] = pd.to_datetime(df['date'])
    df.set_index(['date'])
    df.index = pd.to_datetime(df.date, unit='s')
    
    for itype in interval_types:
        df2 = df.i.resample(itype).sum()
        if len(df2)>num_intervals:
            # this is a good interval, check the number of verbatims per interval
            interval_avg = df2[df2.index].mean()
            if interval_avg < 300:
                print("Average number of documents per interval is low: {}".format(int(interval_avg)))
            return(itype)

    print("Did not find a good range type [M,W,D] for {} intervals. Using D".format(num_intervals))
    return "D"

def last_day_prior_month(dt):
    dt_new = dt.replace(day=1)
    return dt_new - timedelta(days=1)

def get_best_subset_fields(client, metadata=None):
    if metadata == None:
        metadata = client.get('/metadata/')['result']
    field_names = []
    for md in metadata:
        if 'values' in md:
            if len(md['values'])<200:
                field_names.append(md['name'])
            else:
                print("Score driver subsets: Too many values in field_name: {}".format(md['name']))
    return field_names

def get_fieldvalues_for_fieldname(client, field_name, metadata=None):
    if metadata == None:
        metadata = client.get('/metadata/')['result']
    field_names = [d['name'] for d in metadata]
    if field_name in field_names:
        return [[item['value']] for item in [d['values'] for d in metadata if d['name']==field_name][0]]
    else:
        print ("Invalid field name:"+field_name)
        print ("Fieldnames: "+str(field_names))
        return 
    
def create_one_table(client, field, topic_drive, root_url='',filter=""):
    '''
    Create tabulation of ScoreDrivers output, complete with doc counts, example docs, scores and driver clusters
    :param client: LuminosoClient object pointed to project path
    :param driver_fields: List of driver fields (string list)
    :param topic_drive: Whether or not to include saved/top concepts as drivers (bool)
    :return: List of drivers with scores, example docs, clusters and type
    '''
    driver_table = []
    if topic_drive:
        if len(filter)>0:
            score_drivers = client.get('concepts/score_drivers', score_field=field,
                                    concept_selector={'type': 'saved'},filter=filter)
        else:
            score_drivers = client.get('concepts/score_drivers', score_field=field,
                                    concept_selector={'type': 'saved'})
        for driver in score_drivers:
            row = {}
            row['driver'] = driver['name']
            row['type'] = 'saved'
            row['driver_field'] = field
            row['impact'] = driver['impact']
            row['related_terms'] = driver['texts']
            row['doc_count'] = driver['exact_match_count']

            if len(root_url)>0:
                row['url'] = root_url+"/galaxy?suggesting=false&search="+urllib.parse.quote(" ".join(driver['texts']))

            # Use the driver term to find related documents
            search_docs = client.get('docs', search={'texts': driver['texts']}, limit=500, exact_only=True)

            # Sort documents based on their association with the coefficient vector
            for doc in search_docs['result']:
                doc['driver_as'] = get_as(driver['vector'],doc['vector'])

            docs = sorted(search_docs['result'], key=lambda k: k['driver_as']) 
            row['example_doc'] = ''
            row['example_doc2'] = ''
            row['example_doc3'] = ''
            # excel has a max csv cell length of 32767       
            if len(docs) >= 1:
                row['example_doc'] = docs[0]['text'][:32700]
            if len(docs) >= 2:
                row['example_doc2'] = docs[1]['text'][:32700]
            if len(docs) >= 3:
                row['example_doc3'] = docs[2]['text'][:32700]
            driver_table.append(row)


        if len(filter)>0:
            score_drivers = client.get('concepts/score_drivers', score_field=field,
                                    concept_selector={'type': 'top'},filter=filter)            
        else:
            score_drivers = client.get('concepts/score_drivers', score_field=field,
                                    concept_selector={'type': 'top'})
        for driver in score_drivers:
            row = {}
            row['driver'] = driver['name']
            row['type'] = 'top'
            row['driver_field'] = field
            row['impact'] = driver['impact']
            row['related_terms'] = driver['texts']
            row['doc_count'] = driver['exact_match_count']

            if len(root_url)>0:
                row['url'] = root_url+"/galaxy?suggesting=false&search="+urllib.parse.quote(" ".join(driver['texts']))

            # Use the driver term to find related documents
            search_docs = client.get('docs', search={'texts': driver['texts']}, limit=500, exact_only=True)

            # Sort documents based on their association with the coefficient vector
            for doc in search_docs['result']:
                doc['driver_as'] = get_as(driver['vector'],doc['vector'])

            docs = sorted(search_docs['result'], key=lambda k: k['driver_as']) 
            row['example_doc'] = ''
            row['example_doc2'] = ''
            row['example_doc3'] = ''
            # excel has a max csv cell length of 32767
            if len(docs) >= 1:
                row['example_doc'] = docs[0]['text'][:32700]
            if len(docs) >= 2:
                row['example_doc2'] = docs[1]['text'][:32700]
            if len(docs) >= 3:
                row['example_doc3'] = docs[2]['text'][:32700]
            
            driver_table.append(row)

    if len(filter)>0:
        score_drivers = client.get('concepts/score_drivers', score_field=field, limit=100,filter=filter)
    else:
        score_drivers = client.get('concepts/score_drivers', score_field=field, limit=100)
    score_drivers = [d for d in score_drivers if d['importance'] >= .4]
    for driver in score_drivers:
        row = {}
        row['driver'] = driver['name']
        row['type'] = 'auto_found'
        row['driver_field'] = field
        row['impact'] = driver['impact']
        row['related_terms'] = driver['texts']
        row['doc_count'] = driver['exact_match_count']

        if len(root_url)>0:
            row['url'] = root_url+"/galaxy?suggesting=false&search="+urllib.parse.quote(" ".join(driver['texts']))

        # Use the driver term to find related documents
        search_docs = client.get('docs', search={'texts': driver['texts']}, limit=500, exact_only=True)

        # Sort documents based on their association with the coefficient vector
        for doc in search_docs['result']:
            doc['driver_as'] = get_as(driver['vector'],doc['vector'])

        docs = sorted(search_docs['result'], key=lambda k: k['driver_as']) 
        row['example_doc'] = ''
        row['example_doc2'] = ''
        row['example_doc3'] = ''
        # excel has a max csv cell length of 32767
        if len(docs) >= 1:
            row['example_doc'] = docs[0]['text'][:32700]
        if len(docs) >= 2:
            row['example_doc2'] = docs[1]['text'][:32700]
        if len(docs) >= 3:
            row['example_doc3'] = docs[2]['text'][:32700]   
        driver_table.append(row)

    return driver_table

def create_one_sdot_table(client, field, topic_drive, root_url, filter):
    print("{}:{} sdot starting".format(filter[0]['maximum'],field))

    driver_table = create_one_table(client, field, topic_drive, root_url,filter)
    for d in driver_table:
        d['end_date'] = filter[0]['maximum']
    print("{}:{} sdot done data len={}".format(filter[0]['maximum'],field,len(driver_table)))
    return driver_table

def create_drivers_table(client, driver_fields, topic_drive, root_url='',filter="", subset_name=None, subset_value=None):
    all_tables = []
    for field in driver_fields:
        # print("create_driver: {}, {}, {}, {}".format(field, topic_drive, root_url, filter))
        table = create_one_table(client, field, topic_drive, root_url, filter)
        # print("resp_len {}: {}".format(field,len(table)))
        all_tables.extend(table)
    
    if subset_name != None:
        for ti in all_tables:
            ti['subset_name'] = subset_name
            ti['subset_value'] = subset_value

    return all_tables

'''
def create_drivers_with_subsets_table(client, driver_fields, topic_drive, subset_fields=None):

    metadata = client.get('/metadata/')['result']

    # if the user specifies the list of subsets to process
    if subset_fields == None:
        subset_fields = get_best_subset_fields(client, metadata=metadata)
    else:
        subset_fields = subset_fields.split(",")

    # process score drivers by subset
    driver_table = []

    for field_name in subset_fields:
        field_values = get_fieldvalues_for_fieldname(client, field_name, metadata=metadata)
        print("{}: field_values = {}".format(field_name,field_values))
        for field_value in field_values:
            print("score driver for subset [{}:{}]".format(field_name,field_value[0]))
            filter = [{"name":field_name,
                        "values":field_value}]
            sd_data = create_drivers_table(client,driver_fields,topic_drive,filter=filter, subset_name=field_name, subset_value=field_value[0])

            driver_table.extend(sd_data)
            if len(sd_data)>0:
                print("[{}:{}] complete".format(sd_data[0]['subset_name'],sd_data[0]['subset_value']))
            else:
                print("[{}] complete".format(field_name))

    return driver_table
'''

def create_drivers_with_subsets_table(client, driver_fields, topic_drive, root_url='', subset_fields=None):
    futures = []

    metadata = client.get('/metadata/')['result']

    # if the user specifies the list of subsets to process
    if subset_fields == None or len(subset_fields)==0:
        subset_fields = get_best_subset_fields(client, metadata=metadata)
    else:
        subset_fields = subset_fields.split(",")

    # process score drivers by subset
    driver_table = []
    threads_complete = 1
    threads_started = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

        for field_name in subset_fields:
            field_values = get_fieldvalues_for_fieldname(client, field_name, metadata=metadata)
            print("{}: field_values = {}".format(field_name,field_values))
            for field_value in field_values:
                #print("score driver for subset [{}:{}]".format(field_name,field_value[0]))
                filter = [{"name":field_name,
                            "values":field_value}]
                print("filter={}".format(filter))
                futures.append(executor.submit(create_drivers_table,client,driver_fields,topic_drive,root_url=root_url,filter=filter, subset_name=field_name, subset_value=field_value[0])) 
                threads_started += 1

        for future in concurrent.futures.as_completed(futures):
            sd_data = future.result()
            driver_table.extend(sd_data)
            if len(sd_data)>0:
                print("Thread [{} of {}] {}:{} complete. len={}".format(threads_complete,threads_started,sd_data[0]['subset_name'],sd_data[0]['subset_value'],len(sd_data)))
            else:
                print("Thread [{} of {}] complete. len=0".format(threads_complete,threads_started))
            threads_complete += 1
    
    return driver_table

def create_sdot_table(client, driver_fields, date_field_info, end_date, iterations, range_type, topic_drive, root_url='', docs=None):

    sdot_export_data = []
    sd_data_raw = []
    futures = []
    
    if end_date == None or len(end_date)==0:
        end_date = date_field_info['maximum']

    date_field_name = date_field_info['name']
    try:
        end_date_dt = datetime.strptime(end_date, '%m/%d/%Y')
    except:
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ')

    end_date_epoch = end_date_dt.timestamp()
    start_date_dt = None

    if range_type==None or range_type not in ['M','W','D']:
        if docs==None:
            docs = get_all_docs(client)
        range_type = find_best_interval(client,docs,date_field_name,iterations)

    print("sdot threads starting. Date Field: {}, Iterations: {}, Range Type: {}".format(date_field_name, iterations, range_type))
    threads_complete = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

        # run the number of iterations
        for count in range(iterations):
            if range_type=="M":
                if not start_date_dt:
                    if end_date_dt.day==1:
                        print("error, cannot start with the beginning of a month. Starting with previous month")
                        end_date_dt = end_date_dt - timedelta(days=1)
                    start_date_dt = end_date_dt.replace(day=1)
                else:
                    end_date_dt = last_day_prior_month(start_date_dt)
                    start_date_dt = end_date_dt.replace(day=1)

                end_date_epoch = end_date_dt.timestamp()
                start_date_epoch = start_date_dt.timestamp()

            elif range_type=="W":  # week
                start_date_epoch = end_date_epoch - 60*60*24*7
            else:  # day
                start_date_epoch = end_date_epoch - 60*60*24

            # if there is a metadata field filter, apply it here
            for field_value in driver_fields:
                filter = [{"name": date_field_name,
                            "minimum": int(start_date_epoch),
                            "maximum": int(end_date_epoch)}]

                #sd_data_raw.extend(get_score_drivers(proj_apiv5,token,filter))
                futures.append(executor.submit(create_one_sdot_table,client,field_value,topic_drive, root_url,filter)) 

            # move to the nextdate
            end_date_epoch = start_date_epoch
            end_date_dt = datetime.fromtimestamp(end_date_epoch)

        for future in concurrent.futures.as_completed(futures):
            sd_data = future.result()
            sd_data_raw.extend(sd_data)

            print("Thread {} of {} finished".format(threads_complete,iterations))
            threads_complete += 1

    return sd_data_raw

def get_all_docs(client):
    docs = []
    while True:
        new_docs = client.get('docs', limit=25000, offset=len(docs), include_sentiment=True)['result']
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
    parser.add_argument('--topic_drivers', default=False, action='store_true', help="If set, will calculate drivers based on user-defined topics as well")
    parser.add_argument('--encoding', default='utf-8', help="Encoding type of the files to write to")
    parser.add_argument('--sdot', action='store_true', help="Calculate over time")
    parser.add_argument('--sdot_end',default=None, help="Last date to calculat sdot MM/DD/YYYY - algorithm works moving backwards in time.")
    parser.add_argument('--sdot_iterations',default=7, help="Number of over time samples")
    parser.add_argument('--sdot_range',default=None, help="Size of each sample: M,W,D. If none given, range type will be calculated for best fit")
    parser.add_argument('--sdot_date_field',default=None,help="The name of the date field. If none, the first date field will be used")
    parser.add_argument('--subset', default=False, action='store_true', help="Include score drivers by subset")
    parser.add_argument('--subset_fields', default=None, help='Which subsets to include. Default = All with < 200 unique values. Samp: "field1,field2"')
    args = parser.parse_args()
    
    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v5'
    project_id = project_url.split('/')[6].strip()
    workspace_id = project_url.split('/')[5].strip()
    
    client = LuminosoClient.connect(url='%s/projects/%s' % (api_url.strip('/'), project_id))

    #print('Getting Docs...')
    #docs = get_all_docs(client)

    print('Getting Drivers...')
    driver_fields = get_driver_fields(client)
    print("driver_fields={}".format(driver_fields))
    if bool(args.sdot):
        print("Calculating sdot")

        if args.sdot_date_field == None:
            date_field_info = get_first_date_field(client, True)
            if date_field_info == None:
                print("ERROR no date field in project")
                return
        else:
            date_field_info = get_date_field_by_name(args.sdot_date_field)
            if date_field_info == None:
                print("ERROR: no date field name: {}".format(args.sdot_date_field))
                return

        sdot_table = create_sdot_table(client, driver_fields, date_field_info, args.sdot_end, int(args.sdot_iterations), args.sdot_range, args.topic_drivers, root_url='')
        write_table_to_csv(sdot_table, 'sdot_table.csv', encoding=args.encoding)

    driver_table = create_drivers_table(client, driver_fields, args.topic_drivers)
    write_table_to_csv(driver_table, 'drivers_table.csv', encoding=args.encoding)

    # find score drivers by subset
    if bool(args.subset):
        driver_table = create_drivers_with_subsets_table(client, driver_fields, args.topic_drivers,subset_fields=args.subset_fields)
        write_table_to_csv(driver_table, 'subset_drivers_table.csv', encoding=args.encoding)
     
if __name__ == '__main__':
    main()
