
# coding: utf-8

# ## Score Drivers Over Time
# 
# Given an project which includes a date field, and topics you are interested in finding score driver, this find the score driver for each topic contained in the project.
# 
# This script will read all the documents from the project you specify. It reads them in and creates a new project using one weeks' worth of data for a 52 week period. It keeps 16 weeks of documents as it roles towards the final output.
# 
# The input is an account_id/project_id and a project name which is used when writing the output file.
# 
# The documents in the project will need to have predictors. The output is based on a score driver for each topic/predictor combination for the date specified.  For best results the project needs at least 52 weeks of data leading up to the final 16 week window.
# 
# v4 added the use of a branch function to create the new project instead of creating a new project from scratch. This saves doc count.
#
# Sample Command Line:
#    python ScoreDriversOverTimeV4-simplified.py -e 2018-05-25 -a https://eu-daylight.luminoso.com/api/v4 -r 16 -n 17 -o NoName c86f546w prdcj9gb

from luminoso_api import LuminosoClient
import argparse
import datetime, time, json, os, csv
import numpy, pack64



def wait_for_recalculation(client):
    print('Waiting for recalculation')
    counter = 0
    while True:
        time.sleep(15)
        counter += 15
        print('Been waiting {} sec.'.format(counter))
        if not client.get()['running_jobs']:
            break


def main(args):
    account_id = args.account_id  # account id that holds the project
    project_id = args.project_id  # project with all the data
    project_name = args.output_name  # results file will include this name

    print("reading docs from: {}/{}".format(str(account_id),str(project_id)))
    # Get Master Data
    docs = []
    client = LuminosoClient.connect('/projects/{}/{}'.format(account_id,project_id))
    while True:
        new_docs = client.get('docs', limit=25000, offset=len(docs), doc_fields=['text',
                                                                                'date',
                                                                                'predict',
                                                                                'source',
                                                                                'title',
                                                                                'subsets',
                                                                                '_id'])
        if new_docs:
            docs.extend(new_docs)
        else:
            break

    # ## Date window
    # Set the end_date here for the final week you would like to calculate.
    #
    # Given that final week 'end_date', the starting date will be set 'weeks_to_process' weeks
    # in the past and the processing will start there. Given that each week will process 'rolling_weeks'
    # worth of data leading up to and including that week, the first week will be 'rolling weeks' into t
    # he future of that starting date.
    #
    # For instance if I choose, the date May 25, 2018 and set weeks to process to 52 and set
    # rolling_weeks to 16, the starting date will be May 26, 2017, but the first week that get's
    # processed will by (May 5, 2017 plus 16 weeks) which is  September 15, 2017. There will be
    # 52 - 16 = 36 weekly calculated result sets.  Each result set calculates the prior
    # 'rolling_weeks' in this case 16 weeks of data prior to the week being calculated.
    #
    # If you have more sample data each week you can reduce the 'rolling_weeks' or increase
    # 'rolling_weeks' in sparse data sets to have enough language to process.
    #
    #


    end_date=args.end_date
    date_format='%Y-%m-%d'
    end_time_final = int(time.mktime(time.strptime(end_date,date_format)))
    weeks_to_process = int(args.num_weeks)
    rolling_weeks = int(args.rolling_weeks)

    # Sort by date, split by week, run ScoreDrivers
    docs = sorted(docs, key=lambda k: k['date'])
    idx = 0

    end_index = weeks_to_process - rolling_weeks

    # set the initial end_time which will be 16 weeks back
    start_time = end_time_final - (60*60*24*7*weeks_to_process)
    end_time = start_time + (60*60*24*7*rolling_weeks)

    fieldnames=['doc_count','term','text','vector','regressor_dot','driver_score','similar_terms','related_terms',
               'week','predictor', 'date']
    print("opening output file")
    with open('ScoreDriversOverTime{}_results.csv'.format(project_name), 'a') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        while idx<end_index:
            print("start date: {} ({})".format(datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d'),str(start_time)))
            print("end date: {} ({})".format(datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d'),str(start_time)))

            doc_ids = [d['_id'] for d in docs if d['date'] >= start_time and d['date'] < end_time]

            print("starting branch size="+str(len(doc_ids)))
            branch_results = client.post('project/branch/',ids=doc_ids)
            print("finished branch")
            client_branch = LuminosoClient.connect(branch_results['path'])

            wait_for_recalculation(client_branch)
            client_branch.post('prediction/train')
            wait_for_recalculation(client_branch)

            trained_regressors = client_branch.upload('prediction',[{'text':'this is a test'}])[0]
            predictors = list(trained_regressors.keys())
            predictors = list(set(predictors))

            print('Dumping predictor results into file')
            #print('  Predictors: {}'.format(str(predictors)))
            for predictor in predictors:
                drivers = client_branch.put('prediction/drivers', predictor_name=predictor)
                for driver in drivers:
                    # ADDED RELATED TERMS
                    driver['predictor'] = predictor
                    driver['week'] = idx
                    driver['date'] = end_time
                    doc_count = client_branch.get('terms/doc_counts', terms=driver['terms'], use_json=True)
                    count_sum = 0
                    for doc_dict in doc_count:
                        count_sum += (doc_dict['num_exact_matches'] + doc_dict['num_related_matches'])
                    driver['doc_count'] = count_sum
                    if idx == 52:
                        print(driver)

                writer.writerows([{k:v for k,v in d.items() if k in fieldnames} for d in drivers])
                #print('Dumped results to file. Predictor: {}'.format(predictor))

            # delete the project
            if (args.delete):
                print("deleting project")
                delete_result = client_branch.delete()
                print(delete_result)

            end_time = end_time + 60*60*24*7
            start_time = end_time - (60*60*24*7*rolling_weeks)

            idx += 1
    print("DONE")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create inputs to score driver processing'
    )
    parser.add_argument(
        'account_id',
        help="The ID of the account that owns the project, such as 'demo'"
        )
    parser.add_argument(
        'project_id',
        help="The ID of the project"
        )
    parser.add_argument(
        '-u', '--username',
        help='Username (email) of Luminoso account'
        )
    parser.add_argument(
        '-a', '--api_url',
        help='Luminoso API endpoint (https://eu-analytics.luminoso.com/api/v4)'
        )
    parser.add_argument(
        '-e', '--end_date', default=75,
        help="The final week of data in 'YYYY-MM-DD' format (which includes prior rolling_weeks worth of documents)"
        )
    parser.add_argument(
        '-n', '--num_weeks', default=1000,
        help="The total number of weeks to process (output will be reduced by rolling_weeks for data_set)"
        )
    parser.add_argument(
        '-r', '--rolling_weeks', default=1,
        help='The number of weeks to process for each data set of output'
        )
    parser.add_argument(
        '-o', '--output_name', default=1,
        help='Give a name to the output file'
        )
    parser.add_argument(
        '-d', '--delete', action='store_true',
        help='Delete the temporary project'
    )
    args = parser.parse_args()
    main(args)
