# ## Onsite Invite Users
#
# This script has two modes.
#     The first (-c) will read a csv file with the following two columns
#          email,account_id
#       It will send an invite to each email using the account_id which the email address will be associated.  If the email address is already part of that account_id it will be ignored.
#     The second (-a -e -p) will send an invite to a single user.
#

from luminoso_api import V4LuminosoClient as LuminosoClient
import argparse
import csv
import os


# function to invite a user to join a specific account
def invite_user(client, account_id, email, account_permissions):
    users_on_acct = client.get("/{}/users/".format(account_id))
    if email in users_on_acct["members"].keys():
        print("user {} exists - IGNORED".format(email))
    else:
        client.post(
            "/{}/invite/".format(account_id),
            permissions=account_permissions,
            email=email,
        )
        print("email=" + email + "  account=" + account_id + " - INVITED")


def main():

    parser = argparse.ArgumentParser(
        description="Invite users to an onsite installation. "
        + "There are two ways to run this script using -c with csv list of email address "
        + "or using -a -e -p options to add a single user."
    )

    parser.add_argument(
        "host_url",
        help="Luminoso API endpoint (e.g., https://daylight.luminoso.com)",
    )

    csv_group = parser.add_argument_group(title='arguments for using a CSV')
    csv_group.add_argument(
        "-c",
        "--csv_file",
        help="The csv file that holds rows [email,account_id,permissions]."
    )
    csv_group.add_argument(
        "--encoding", default="utf-8-sig", help="Encoding type of the file to read from"
    )

    single_group = parser.add_argument_group(title='arguments for adding a single user')
    single_group.add_argument(
        "-a", "--account_id", help="Account ID to add the user under"
    )
    single_group.add_argument(
        "-e",
        "--email",
        help="Email address to user for the account name and invite. User will receive an email explaining how to setup an account",
    )
    single_group.add_argument(
        "-p",
        "--permissions",
        help='Permission(s) to give an individual user comma separated: "read,write,create"',
    )
    args = parser.parse_args()

    csv_file = args.csv_file
    account_id = args.account_id
    email = args.email
    permissions = args.permissions

    client = LuminosoClient.connect(args.host_url, user_agent_suffix='se_code:onsite_invite_user')
    accounts_client = client.change_path("/accounts/")

    if all(arg is None for arg in (csv_file, account_id, email, permissions)):
        parser.error('You must specify either a CSV file or an account ID,'
                     ' email, and permissions.')

    if csv_file is not None:
        if account_id is not None or email is not None or permissions is not None:
            parser.error('When specifying a CSV file, the options for adding'
                         ' a single user cannot be specified.')
        with open(csv_file, encoding=args.encoding) as f:
            table = list(csv.DictReader(f))
        for acct in table:
            perm = acct["permissions"].split(",")
            invite_user(accounts_client, acct["account_id"], acct["email"],
                        perm)

    else:
        if account_id is None or email is None or permissions is None:
            parser.error('When inviting a single user, account ID and email'
                         ' and permissions must all be specified.')
        perm = permissions.split(",")
        invite_user(accounts_client, account_id, email, perm)


if __name__ == "__main__":

    main()
