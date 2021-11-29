import argparse
import csv
import numpy as np
import pandas as pd
import urllib.parse
from datetime import datetime, timedelta

from luminoso_api import V5LuminosoClient as LuminosoClient
from pack64 import unpack64

DOC_BATCH_SIZE = 5000


class LuminosoData:
    def __init__(self, client, root_url=''):
        self.client = client
        self.root_url = root_url
        self._docs = None
        self._metadata = None

    def set_root_url(self, root_url):
        """
        root_url can (and perhaps should) be set on initialization, but to
        simplify matters for its use in pull_lumi_data, we make it possible to
        set it after.
        """
        self.root_url = root_url

    @property
    def docs(self):
        if self._docs is None:
            self._docs = []
            while True:
                new_docs = self.client.get(
                    'docs', limit=DOC_BATCH_SIZE, offset=len(self._docs),
                    include_sentiment_on_concepts=True
                )['result']
                if new_docs:
                    self._docs.extend(new_docs)
                else:
                    break
        return self._docs

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.client.get('metadata')['result']
        return self._metadata

    @property
    def driver_fields(self):
        '''
        Get all numeric or score metadata fields from the project in order to
        run drivers against
        :return: List of fields that contain drivers
        '''
        driver_fields = [m['name'] for m in self.metadata
                         if m['type'] == 'number' or m['type'] == 'score']
        return driver_fields

    @property
    def first_date_field(self):
        '''
        Get the first date field
        :return: dictionary with the date field info
        '''
        date_fields = [df for df in self.metadata if df['type'] == 'date']
        if len(date_fields) > 1:
            print("WARNING: multiple date fields. Using first date field"
                  " found.")
        if not date_fields:
            return None
        return date_fields[0]

    def get_fieldvalues_for_fieldname(self, field_name):
        field = self.get_field_by_name(field_name)
        if not field:
            print("Invalid field name:", field_name)
            print("Fieldnames:", [d['name'] for d in self.metadata])
            return
        return [[item['value']] for item in field['values']]

    def get_field_by_name(self, field_name):
        '''
        Get a field by name
        :param field_name: name of the date field to get
        :return: dictionary with the field info, or `None` if no field with
          that name exists
        '''
        for field in self.metadata:
            if field['name'] == field_name:
                return field
        return None

    def get_best_subset_fields(self):
        field_names = []
        for md in self.metadata:
            if 'values' in md:
                if len(md['values']) < 200:
                    field_names.append(md['name'])
                else:
                    print(
                        "Score driver subsets: Too many values in field_name:"
                        " {}".format(md['name']))
        return field_names

    def find_best_interval(self, date_field_name, num_intervals):
        docs_by_date = []
        for i, d in enumerate(self.docs):
            for m in d['metadata']:
                if m['name'] == date_field_name:
                    date = datetime.strptime(m['value'],
                                             '%Y-%m-%dT%H:%M:%S.%fZ')
                    docs_by_date.append({'date': date, 'doc_id': d['doc_id'],
                                         'i': 1})
                    break

        df = pd.DataFrame(docs_by_date)
        df.set_index(['date'])
        pd.to_datetime(df.date, unit='s')

        interval_types = ['M', 'W', 'D']
        df = pd.DataFrame(docs_by_date)
        df.set_index(['date'])
        df.index = pd.to_datetime(df.date, unit='s')

        for itype in interval_types:
            df2 = df.i.resample(itype).sum()
            if len(df2) > num_intervals:
                # this is a good interval, check the number of verbatims per
                # interval
                interval_avg = df2[df2.index].mean()
                if interval_avg < 300:
                    print("Average number of documents per interval is low:"
                          " {}".format(int(interval_avg)))
                return itype

        print("Did not find a good range type [M,W,D] for {} intervals."
              " Using D".format(num_intervals))
        return "D"


def get_assoc(vector1, vector2):
    '''
    Calculate the association score between two vectors
    :param vector1: First vector
    :param vector2: Second vector
    :return: Cosine similarity of two vectors
    '''
    return float(np.dot(unpack64(vector1), unpack64(vector2)))


def get_driver_url(root_url, driver):
    texts = urllib.parse.quote(' '.join(driver['texts']))
    return root_url + '/galaxy?suggesting=false&search=' + texts


def create_one_table(client, field, topic_drive, root_url='', filter_list=""):
    '''
    Create tabulation of ScoreDrivers output, complete with doc counts, example
    docs, scores and driver clusters
    :param client: LuminosoClient object pointed to project path
    :param driver_fields: List of driver fields (string list)
    :param topic_drive: Whether or not to include saved/top concepts as drivers (bool)
    :return: List of drivers with scores, example docs, clusters and type
    '''
    driver_table = []
    if topic_drive:
        if len(filter_list) > 0:
            score_drivers = client.get(
                'concepts/score_drivers', score_field=field,
                concept_selector={'type': 'saved'}, filter=filter_list
            )
        else:
            score_drivers = client.get(
                'concepts/score_drivers', score_field=field,
                concept_selector={'type': 'saved'}
            )
        for driver in score_drivers:
            row = {'driver': driver['name'], 'type': 'saved',
                   'driver_field': field, 'impact': driver['impact'],
                   'related_terms': driver['texts'],
                   'doc_count': driver['exact_match_count']}

            if len(root_url) > 0:
                row['url'] = get_driver_url(root_url, driver)

            # Use the driver term to find related documents
            search_docs = client.get('docs', search={'texts': driver['texts']},
                                     limit=500, match_type='exact')

            # Sort documents based on their association with the coefficient
            # vector
            for doc in search_docs['result']:
                doc['driver_as'] = get_assoc(driver['vector'], doc['vector'])

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

        if len(filter_list) > 0:
            score_drivers = client.get(
                'concepts/score_drivers', score_field=field,
                concept_selector={'type': 'top'}, filter=filter_list
            )
        else:
            score_drivers = client.get(
                'concepts/score_drivers', score_field=field,
                concept_selector={'type': 'top'}
            )
        for driver in score_drivers:
            row = {'driver': driver['name'], 'type': 'top',
                   'driver_field': field, 'impact': driver['impact'],
                   'related_terms': driver['texts'],
                   'doc_count': driver['exact_match_count']}

            if len(root_url) > 0:
                row['url'] = get_driver_url(root_url, driver)

            # Use the driver term to find related documents
            search_docs = client.get('docs', search={'texts': driver['texts']},
                                     limit=500, match_type='exact')

            # Sort documents based on their association with the coefficient
            # vector
            for doc in search_docs['result']:
                doc['driver_as'] = get_assoc(driver['vector'], doc['vector'])

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

    if len(filter_list) > 0:
        score_drivers = client.get('concepts/score_drivers', score_field=field,
                                   limit=100, filter=filter_list)
    else:
        score_drivers = client.get('concepts/score_drivers', score_field=field,
                                   limit=100)
    score_drivers = [d for d in score_drivers if d['importance'] >= .4]
    for driver in score_drivers:
        row = {'driver': driver['name'], 'type': 'auto_found',
               'driver_field': field, 'impact': driver['impact'],
               'related_terms': driver['texts'],
               'doc_count': driver['exact_match_count']}

        if len(root_url) > 0:
            row['url'] = get_driver_url(root_url, driver)

        # Use the driver term to find related documents
        search_docs = client.get('docs', search={'texts': driver['texts']},
                                 limit=500, match_type='exact')

        # Sort documents based on their association with the coefficient vector
        for doc in search_docs['result']:
            doc['driver_as'] = get_assoc(driver['vectors'][0], doc['vector'])

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


def create_one_sdot_table(client, field, topic_drive, root_url, filter_list):
    print("{}:{} sdot starting".format(filter_list[0]['maximum'], field))

    driver_table = create_one_table(client, field, topic_drive, root_url,
                                    filter_list)
    for d in driver_table:
        d['end_date'] = filter_list[0]['maximum']
    print("{}:{} sdot done data len={}".format(
        filter_list[0]['maximum'], field, len(driver_table))
    )
    return driver_table


def create_drivers_table(luminoso_data, topic_drive, root_url='',
                         filter_list="", subset_name=None, subset_value=None):
    all_tables = []
    for field in luminoso_data.driver_fields:
        table = create_one_table(luminoso_data.client, field, topic_drive,
                                 root_url, filter_list)
        all_tables.extend(table)

    if subset_name is not None:
        for ti in all_tables:
            ti['subset_name'] = subset_name
            ti['subset_value'] = subset_value

    return all_tables


def create_drivers_with_subsets_table(luminoso_data, topic_drive,
                                      root_url='', subset_fields=None):
    # if the user specifies the list of subsets to process
    if not subset_fields:
        subset_fields = luminoso_data.get_best_subset_fields()
    else:
        subset_fields = subset_fields.split(",")

    # process score drivers by subset
    driver_table = []

    for field_name in subset_fields:
        field_values = luminoso_data.get_fieldvalues_for_fieldname(field_name)
        print("{}: field_values = {}".format(field_name, field_values))
        for field_value in field_values:
            filter_list = [{"name": field_name, "values": field_value}]
            print("filter={}".format(filter_list))
            sd_data = create_drivers_table(
                luminoso_data, topic_drive, root_url=root_url,
                filter_list=filter_list, subset_name=field_name,
                subset_value=field_value[0]
            )
            driver_table.extend(sd_data)
            if len(sd_data) > 0:
                print("{}:{} complete. len={}".format(
                    sd_data[0]['subset_name'], sd_data[0]['subset_value'],
                    len(sd_data)
                ))

    return driver_table


def create_sdot_table(luminoso_data, date_field_info, end_date, iterations,
                      range_type, topic_drive, root_url=''):
    sd_data_raw = []

    if end_date is None or len(end_date) == 0:
        end_date = date_field_info['maximum']

    date_field_name = date_field_info['name']
    try:
        end_date_dt = datetime.strptime(end_date, '%m/%d/%Y')
    except ValueError:
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ')

    end_date_epoch = end_date_dt.timestamp()
    start_date_dt = None

    if range_type is None or range_type not in ['M', 'W', 'D']:
        range_type = luminoso_data.find_best_interval(date_field_name,
                                                      iterations)

    print("sdot threads starting. Date Field: {}, Iterations: {},"
          " Range Type: {}".format(date_field_name, iterations, range_type))

    # run the number of iterations
    for count in range(iterations):
        if range_type == "M":
            if not start_date_dt:
                if end_date_dt.day == 1:
                    print("error, cannot start with the beginning of a"
                          " month. Starting with previous month")
                    end_date_dt = end_date_dt - timedelta(days=1)
                start_date_dt = end_date_dt.replace(day=1)
            else:
                end_date_dt = start_date_dt.replace(day=1) - timedelta(days=1)
                start_date_dt = end_date_dt.replace(day=1)

            end_date_epoch = end_date_dt.timestamp()
            start_date_epoch = start_date_dt.timestamp()

        elif range_type == "W":  # week
            start_date_epoch = end_date_epoch - 60*60*24*7
        else:  # day
            start_date_epoch = end_date_epoch - 60*60*24

        # if there is a metadata field filter, apply it here
        for field_value in luminoso_data.driver_fields:
            filter_list = [{"name": date_field_name,
                            "minimum": int(start_date_epoch),
                            "maximum": int(end_date_epoch)}]

            sd_data = create_one_sdot_table(luminoso_data.client, field_value,
                                            topic_drive, root_url, filter_list)
            sd_data_raw.extend(sd_data)

        # move to the nextdate
        end_date_epoch = start_date_epoch
        end_date_dt = datetime.fromtimestamp(end_date_epoch)

    return sd_data_raw


def write_table_to_csv(table, filename, encoding='utf-8'):
    '''
    Function for writing lists of dictionaries to a CSV file
    :param table: List of dictionaries to be written
    :param filename: Filename to be written to (string)
    :param encoding: File encoding (default utf-8)
    :return: None
    '''
    print('Writing to file {}.'.format(filename))
    if len(table) == 0:
        print('Warning: No data to write to {}.'.format(filename))
        return
    # Get the names of all the fields in all the dictionaries in the table.  We
    # want a set rather then a list--but Python sets don't respect ordering,
    # and we want to keep the columns in the same order as much as possible,
    # so we put them into a dictionary with dummy values.
    fieldnames = {k: None for t_item in table for k in t_item}
    with open(filename, 'w', encoding=encoding, newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(
        description='Export Subset Key Terms and write to a file'
    )
    parser.add_argument('project_url',
                        help="The complete URL of the Daylight project")
    parser.add_argument('--topic_drivers', default=False, action='store_true',
                        help="If set, will calculate drivers based on"
                             " user-defined topics as well")
    parser.add_argument('--encoding', default='utf-8',
                        help="Encoding type of the files to write to")
    parser.add_argument('--sdot', action='store_true',
                        help="Calculate over time")
    parser.add_argument('--sdot_end', default=None,
                        help="Last date to calculate sdot MM/DD/YYYY -"
                             " algorithm works moving backwards in time.")
    parser.add_argument('--sdot_iterations', default=7,
                        help="Number of over time samples")
    parser.add_argument('--sdot_range', default=None,
                        help="Size of each sample: M,W,D. If none given, range"
                             " type will be calculated for best fit")
    parser.add_argument('--sdot_date_field', default=None,
                        help="The name of the date field. If none, the first"
                             " date field will be used")
    parser.add_argument('--subset', default=False, action='store_true',
                        help="Include score drivers by subset")
    parser.add_argument('--subset_fields', default=None,
                        help='Which subsets to include. Default = All with'
                             ' < 200 unique values. Samp: "field1,field2"')
    args = parser.parse_args()

    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v5'
    project_id = project_url.split('/')[6].strip()

    client = LuminosoClient.connect(
        url='%s/projects/%s' % (api_url.strip('/'), project_id),
        user_agent_suffix='se_code:score_drivers'
    )
    luminoso_data = LuminosoData(client)
    print('Getting Drivers...')
    driver_fields = luminoso_data.driver_fields
    print("driver_fields={}".format(driver_fields))
    if bool(args.sdot):
        print("Calculating sdot")

        if args.sdot_date_field is None:
            date_field_info = luminoso_data.first_date_field
            if date_field_info is None:
                print("ERROR no date field in project")
                return
        else:
            date_field_info = luminoso_data.get_field_by_name(
                args.sdot_date_field
            )
            if date_field_info is None:
                print("ERROR: no date field name:"
                      " {}".format(args.sdot_date_field))
                return

        sdot_table = create_sdot_table(
            luminoso_data, date_field_info, args.sdot_end,
            int(args.sdot_iterations), args.sdot_range, args.topic_drivers,
            root_url=''
        )
        write_table_to_csv(sdot_table, 'sdot_table.csv',
                           encoding=args.encoding)

    driver_table = create_drivers_table(luminoso_data, args.topic_drivers)
    write_table_to_csv(driver_table, 'drivers_table.csv',
                       encoding=args.encoding)

    # find score drivers by subset
    if bool(args.subset):
        driver_table = create_drivers_with_subsets_table(
            luminoso_data, args.topic_drivers,
            subset_fields=args.subset_fields
        )
        write_table_to_csv(driver_table, 'subset_drivers_table.csv',
                           encoding=args.encoding)


if __name__ == '__main__':
    main()
