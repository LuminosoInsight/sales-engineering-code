# ## Onsite Usage
#
# This script lists the usage for all the accounts on a Luminoso Daylight
# onsite image.
#

from luminoso_api import V4LuminosoClient as LuminosoClient
import argparse
import csv
import os


# function to list usage on all accounts
def get_usage_list(client):
    usage_csv = []
    results = client.get("/accounts/")
    for a in results["accounts"]:

        # get the usage data
        usage = client.get("/accounts/{}/usage".format(a["account_id"]))

        # add it to a list for export
        usage_csv.extend(
            [
                {
                    "account_id": a["account_id"],
                    "account_name": a["account_name"],
                    "email": a["email"],
                    "docs_uploaded": u["docs_uploaded"],
                    "month": u["month"],
                    "year": u["year"],
                }
                for u in usage["months"]
            ]
        )
    return usage_csv


def main():

    parser = argparse.ArgumentParser(description="List usage on an onsite image ")

    parser.add_argument(
        "-t",
        "--token",
        help="The API token used to access the host. Or use environment variable LUMINOSO_TOKEN",
        default=None,
    )

    parser.add_argument(
        "-u", "--host_url", help="Luminoso host (https://daylight.luminoso.com/)"
    )

    parser.add_argument(
        "-o",
        "--output_file",
        help="The name output file to use [default=onsite_usage_output.csv]",
        default="onsite_usage_output.csv",
    )
    args = parser.parse_args()

    # process the token from either command line, env or tokens.json
    token = args.token
    if not token:
        token = None
    if "LUMINOSO_TOKEN" in os.environ:
        token = os.environ["LUMINOSO_TOKEN"]

    # connect to the Luminoso Daylight onsite service
    api_url = args.host_url + "/api/v4/"
    client = LuminosoClient.connect(api_url, token=token)

    # list the accounts
    usage_csv = get_usage_list(client)

    # send the output to a file
    fields = usage_csv[0].keys()
    with open(args.output_file, "w") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(usage_csv)


if __name__ == "__main__":
    main()
