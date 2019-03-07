
# ## Onsite Invite Users
#
# This script has two modes.
#     The first (-c) will read a csv file with the following two columns
#          email,account_id
#       It will send an invite to each email using the account_id which the email address will be associated.  If the email address is already part of that account_id it will be ignored.
#     The second (-a -e -p) will send an invite to a single user.
#

from luminoso_api import LuminosoClient
import argparse
import csv

# function to invite a user to join a specific account
def invite_user(client,account_id,email,account_permissions):
    users_on_acct = client.get('/{}/users/'.format(account_id))
    if (email in users_on_acct['members'].keys()):
        print('user {} exists - IGNORED'.format(email))
    else:
        client.post('/{}/invite/'.format(account_id),
                email=email,
                permissions=account_permissions)
        print("email="+email+"  account="+account_id+" - INVITED")


def main(args):
    # list of argument keys to use to verify proper usage
    arglist = [k for k,i in vars(args).items() if i!=None]

    # there are two ways to run this script, in batch with a csv file
    # which is this option below
    if args.csv_file != None:
        input_data = './onsite_add_users.csv'

        if any(key in arglist for key in ['account_id','email','permissions']):
            print("WARNING: Single user invite args -a,-e,-p ignored for csv read")

        with open(input_data, encoding=args.encoding) as f:
            reader = csv.DictReader(f)
            table = [row for row in reader]

        # connect to the Luminoso Daylight onsite service
        client = LuminosoClient.connect(args.api_url+'accounts/',token=args.token)

        # iterate the csv and invite each user
        for acct in table:
            perm = acct['permissions'].split(",")
            invite_user(client,acct['account_id'],acct['email'],perm)
    else:
        # option two, invite a single user

        # make sure all three options are specified (-a -e -p)
        if not all(key in arglist for key in ['account_id','email','permissions']):
            print("ERROR: to invite a single user, you must have all three options -a -e -p")

        # connect to the Luminoso Daylight onsite service
        client = LuminosoClient.connect(args.api_url+'accounts/',token=args.token)

        # invite the user
        perm = args.permissions.split(",")
        invite_user(client,args.account_id,args.email,perm)

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
        '-c', '--csv_file',
        help='The csv file that holds rows [email,account_id,permissions]. Use this to add multiple users or -u,-p,-a for an individual user'
        )
    parser.add_argument(
        '-a', '--account_id',
        help='Account ID to add the account under'
        )
    parser.add_argument(
        '-e', '--email',
        help='Email address to user for the account name and invite. User will receive an email explaining how to setup an account'
        )
    parser.add_argument(
        '-p', '--permissions',
        help='Permission(s) to give an individual user comma separated: "read,write,create"'
        )
    parser.add_argument(
        '--encoding', default="utf-8-sig",
        help="Encoding type of the file to read from"
    )
    args = parser.parse_args()
    main(args)
