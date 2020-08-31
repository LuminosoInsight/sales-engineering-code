# ## Onsite List Users
#
# This script lists the users on an onsite image for a given account id
#

from luminoso_api import LuminosoClient
import argparse
import csv
import json

# function to list users
def list_users(client, account_id):
    users_on_acct = client.get("/accounts/{}/users/".format(account_id))["result"]
    print(users_on_acct)
    print()
    all_users = users_on_acct["guests"]
    for uname, udata in all_users.items():
        print("{}:  {}".format(uname, udata))

    return users_on_acct


def main():
    parser = argparse.ArgumentParser(
        description="List users on an onsite installation account."
    )
    parser.add_argument("token", help="The API token used to access the host")

    parser.add_argument(
        "-u",
        "--host_url",
        help="Luminoso API endpoint (https://analytics.luminoso.com)",
        required=True,
    )
    parser.add_argument("-a", "--account_id", help="Account ID to list the users under")
    parser.add_argument(
        "-o",
        "--output_file",
        help="The json output file to use [example=onsite_usage_output.json]",
    )

    args = parser.parse_args()

    api_v4 = args.host_url + "/api/v4/"

    # connect to the Luminoso Daylight onsite service
    client = LuminosoClient.connect(api_v4, token=args.token)

    # list the users
    users_on_account = list_users(client, args.account_id)

    # send the output to a file
    if args.output_file:
        with open(args.output_file, "w") as outfile:
            json.dump(users_on_account, outfile, indent=2)


if __name__ == "__main__":
    main()
