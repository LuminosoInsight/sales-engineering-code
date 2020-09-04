# ## Onsite List Users
#
# This script lists the users on an onsite image for a given account id
#
from luminoso_api import V4LuminosoClient as LuminosoClient
import argparse
import json
import os


# function to list users
def list_users(client, account_id):
    users_on_acct = client.get("/accounts/{}/users/".format(account_id))
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
    parser.add_argument(
        "host_url",
        help="Luminoso host (https://analytics.luminoso.com)",
    )
    parser.add_argument("account_id", help="Account ID to list the users under")
    parser.add_argument(
        "-o",
        "--output_file",
        help="The json output file to use [example=onsite_usage_output.json]",
    )

    args = parser.parse_args()

    # connect to the Luminoso Daylight onsite service
    api_v4 = args.host_url + "/api/v4/"
    client = LuminosoClient.connect(api_v4)

    # list the users
    users_on_account = list_users(client, args.account_id)

    # send the output to a file
    if args.output_file:
        with open(args.output_file, "w") as outfile:
            json.dump(users_on_account, outfile, indent=2)


if __name__ == "__main__":
    main()
