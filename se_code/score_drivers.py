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
        self.cache_docs = {}
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

    def get_galaxy_url_from_concept(self, concept):
        """
        Given a concept as returned by the API (i.e., a dict that contains a
        list of texts in its "texts" field), return the URL for a search on
        those texts in the Galaxy view.  Returns None if there is no root_url.
        """
        if not self.root_url:
            return None
        texts = urllib.parse.quote(' '.join(concept['texts']))
        return self.root_url + '/galaxy?suggesting=false&search=' + texts


def get_assoc(vector1, vector2):
    '''
    Calculate the association score between two vectors
    :param vector1: First vector
    :param vector2: Second vector
    :return: Cosine similarity of two vectors
    '''
    return float(np.dot(unpack64(vector1), unpack64(vector2)))


def _create_rows_from_drivers(luminoso_data, field, api_params, driver_type):
    """
    Helper function for create_one_table().
    """
    rows = []
    score_drivers = luminoso_data.client.get(
        'concepts/score_drivers', score_field=field, **api_params
    )
    if 'concept_selector' not in api_params:
        score_drivers = [d for d in score_drivers if d['importance'] >= .4]

    for driver in score_drivers:
        row = {'driver': driver['name'], 'type': driver_type,
               'driver_field': field, 'impact': driver['impact'],
               'related_terms': driver['texts'],
               'doc_count': driver['exact_match_count']}

        url = luminoso_data.get_galaxy_url_from_concept(driver)
        if url:
            row['url'] = url

        texts_key = "_".join(driver['texts'])
        if texts_key not in luminoso_data.cache_docs:
            # Use the driver term to find related documents
            docs = luminoso_data.client.get(
                'docs', search={'texts': driver['texts']}, limit=500,
                match_type='exact', fields=('text', 'vector')
            )['result']

            # Sort documents based on their association with the coefficient
            # vector
            for doc in docs:
                doc['driver_as'] = get_assoc(driver['vectors'][0], doc['vector'])
            docs.sort(key=lambda k: k['driver_as'], reverse=True)

            luminoso_data.cache_docs[texts_key] = docs[0:3]

        row['example_doc'] = ''
        row['example_doc2'] = ''
        row['example_doc3'] = ''
        # excel has a max csv cell length of 32767
        if len(luminoso_data.cache_docs[texts_key]) >= 1:
            row['example_doc'] = luminoso_data.cache_docs[texts_key][0]['text'][:32700]
        if len(luminoso_data.cache_docs[texts_key]) >= 2:
            row['example_doc2'] = luminoso_data.cache_docs[texts_key][1]['text'][:32700]
        if len(luminoso_data.cache_docs[texts_key]) >= 3:
            row['example_doc3'] = luminoso_data.cache_docs[texts_key][2]['text'][:32700]
        rows.append(row)
    return rows


def create_one_table(luminoso_data, field, topic_drive, filter_list=None):
    '''
    Create tabulation of ScoreDrivers output, complete with doc counts, example
    docs, scores and driver clusters
    :param luminoso_data: a LuminosoData object
    :param field: string representing the score field to use
    :param topic_drive: Whether or not to include saved/top concepts as drivers (bool)
    :param filter_list: document filter (as a list of dicts)
    :return: List of drivers with scores, example docs, clusters and type
    '''
    driver_table = []

    # Set up a dict that's either empty or contains "filter", so we can use it
    # in keyword arguments to client.get()
    api_params = {}
    if filter_list:
        api_params['filter'] = filter_list
    if topic_drive:
        saved_params = dict(api_params, concept_selector={'type': 'saved'})
        driver_table.extend(_create_rows_from_drivers(
            luminoso_data, field, saved_params, 'saved'
        ))

        top_params = dict(api_params, concept_selector={'type': 'top'})
        driver_table.extend(_create_rows_from_drivers(
            luminoso_data, field, top_params, 'top'
        ))

    driver_params = dict(api_params, limit=100)
    driver_table.extend(_create_rows_from_drivers(
        luminoso_data, field, driver_params, 'auto_found'
    ))

    return driver_table


def create_one_sdot_table(luminoso_data, field, topic_drive, filter_list):
    print("{}:{} sdot starting".format(filter_list[0]['maximum'], field))

    driver_table = create_one_table(luminoso_data, field, topic_drive,
                                    filter_list=filter_list)
    for d in driver_table:
        d['end_date'] = filter_list[0]['maximum']
    print("{}:{} sdot done data len={}".format(
        filter_list[0]['maximum'], field, len(driver_table))
    )
    return driver_table


def create_drivers_table(luminoso_data, topic_drive, filter_list=None,
                         subset_name=None, subset_value=None):
    all_tables = []
    for field in luminoso_data.driver_fields:
        table = create_one_table(luminoso_data, field, topic_drive,
                                 filter_list=filter_list)
        all_tables.extend(table)

    if subset_name is not None:
        for ti in all_tables:
            ti['subset_name'] = subset_name
            ti['subset_value'] = subset_value

    return all_tables


def create_drivers_with_subsets_table(luminoso_data, topic_drive,
                                      subset_fields=None):
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
                luminoso_data, topic_drive,
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
                      range_type, topic_drive):
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

            sd_data = create_one_sdot_table(luminoso_data, field_value,
                                            topic_drive, filter_list)
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
            int(args.sdot_iterations), args.sdot_range, args.topic_drivers
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
