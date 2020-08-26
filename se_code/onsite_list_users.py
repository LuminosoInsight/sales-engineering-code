
# ## Onsite List Users
#
# This script lists the users on the system
#

from luminoso_api import LuminosoClient
import argparse
import csv
import json

# function to list users
def list_users(client,account_id):
    users_on_acct = client.get('/accounts/{}/users/'.format(account_id))['result']
    print(users_on_acct)
    print()
    all_users = users_on_acct['guests']
    for uname,udata in all_users.items():
        print('{}:  {}'.format(uname,udata))

# function to list usage on all accounts
def list_usage(client,token):
    usage_csv = []
    results = client.get('/accounts/')['result']
    for a in results['accounts']:

        # get the usage data
        usage = client.get('/accounts/{}/usage'.format(a['account_id']))

        # print(json.dumps(usage,indent=2))

        # add it to a list for export
        usage_csv.extend([ {'account_id':a['account_id'],
                        'account_name':a['account_name'],
                        'email':a['email'],
                        'docs_uploaded':u['docs_uploaded'],
                        'month':u['month'],
                        'year':u['year']} for u in usage['result']['months']])
    # print(json.dumps(usage_csv,indent=2))
    return usage_csv

def main(args):
    # list of argument keys to use to verify proper usage
    arglist = [k for k,i in vars(args).items() if i!=None]

    # make sure all three options are specified (-a -e -p)
    if not all(key in arglist for key in ['account_id']):
        print("ERROR: to list users you must have the option -a")

    # connect to the Luminoso Daylight onsite service
    client = LuminosoClient.connect(args.api_url,token=args.token)

    # list the users
    list_users(client,args.account_id)

    #list the accounts
    # usage_csv = list_usage(client,args.token)

    # send the output to a file
    '''fields = usage_csv[0].keys()
    with open(args.output_file, 'w') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(usage_csv)
    '''
    print("DONE")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Invite users to an onsite installation. '+
            'There are two ways to run this script using -c with csv list of email address '+
            'or using -a -e -p options to add a single user.'
    )
    parser.add_argument(
        'token',
        help="The API token used to access the host"
        )

    parser.add_argument(
        '-u', '--api_url',
        help='Luminoso API endpoint (https://eu-analytics.luminoso.com/api/v4)'
        )
    parser.add_argument(
        '-a', '--account_id',
        help='Account ID to list the users under'
        )
    parser.add_argument(
        '-o', '--output_file',
        help="The name output file to use [default=onsite_usage_output.csv]",
        default='onsite_usage_output.csv'
    )
    args = parser.parse_args()
    main(args)
