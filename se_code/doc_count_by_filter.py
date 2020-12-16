from luminoso_api import V5LuminosoClient as LuminosoClient
import argparse
import csv


def doc_count_by_concept_lists(
    client, concept_lists, match_type="both", field_names=None, field_limit=20
):
    """
    Create a tabulation of doc ids related to concepts. Store results in concept_lists in metadata format
    :param client: LuminosoClient object set to project path
    :param concept_lists: List of concepts (format from /concept_lists endpoint)
    :param match_type: Type of document search ('conceptual', 'exact', 'both') default = both
    :param field_names: List of metadata field names to process. By default use None which means ALL
    :param field_limit: The limit for field values. If there are more than limit then ignore field.
    :return: concept_lists with added metadata field that includes the doc_id lists and counts for each concept
    """
    for cl in concept_lists:
        md = client.get("metadata")["result"]
        cl["metadata"] = md  # storing the results in md for this concept list
        for conceptnum, concept in enumerate(cl["concepts"]):

            for m in md:
                if (not field_names) or (m["name"] in field_names):
                    # can only process metadata items that have lists of values
                    # TODO: date fields can be processed by hour/day/week/month with some division count
                    if "values" in m and len(m["values"]) < field_limit:
                        # give status since this can take a while. It's nice to see
                        # the list of field values, but they might be long so
                        # logic here to show them only once per concept list
                        if conceptnum == 0:
                            print(
                                "{}-{}-{}: values: {}".format(
                                    cl["name"],
                                    concept["name"],
                                    m["name"],
                                    [fv["value"] for fv in m["values"]],
                                )
                            )
                        else:
                            print(
                                "{}-{}-{}".format(
                                    cl["name"], concept["name"], m["name"]
                                )
                            )

                        for fv in m["values"]:

                            if "concept_doc_ids" not in fv:
                                fv["concept_doc_ids"] = {}
                                fv["concept_doc_counts"] = {}
                            more_docs = True
                            docs = []
                            filter = [{"name": m["name"], "values": [fv["value"]]}]
                            while more_docs:

                                new_docs = client.get(
                                    "docs",
                                    search={"texts": concept["texts"]},
                                    filter=filter,
                                    fields=["doc_id", "match_score"],
                                    limit=5000,
                                    offset=len(docs),
                                    match_type=match_type,
                                )

                                if new_docs["result"]:
                                    docs.extend(
                                        {
                                            "doc_id": d["doc_id"],
                                            "match_score": d["match_score"],
                                        }
                                        for d in new_docs["result"]
                                    )
                                else:
                                    more_docs = False
                            fv["concept_doc_counts"][concept["name"]] = len(docs)
                    elif "values" in m:
                        # we might be skipping this field multiple times. only error once
                        # show them how many values there are so the can increase
                        # the limit if they want to.
                        if "have_shown_limit" not in m:
                            print(
                                "Ignoring field: {}. {} field values is over {} limit".format(
                                    m["name"], len(m["values"]), field_limit
                                )
                            )
                            m["have_shown_limit"] = True
                    else:
                        # we might be skipping this field multiple times. only error once
                        if "have_shown_no_values" not in m:
                            print(
                                "Ignoring field: {}. Type {} does not have specific value list.".format(
                                    m["name"], m["type"]
                                )
                            )
                            m["have_shown_no_values"] = True


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the doc counts for each set of concept lists vs. metadata field values"
    )
    parser.add_argument("project_url", help="The URL of the project to analyze")
    parser.add_argument(
        "output_filename", help="The csv file name to save the results in"
    )
    parser.add_argument(
        "-l",
        "--list_names",
        default=None,
        help="The names of this shared concept lists separated by |. Default = ALL lists",
    )
    parser.add_argument(
        "-f",
        "--field_names",
        default=None,
        help="Which metadata field names to compare. Separate multiple values with |. Default=ALL fields with value lists and number of values less than field_limit parameter",
    )
    parser.add_argument(
        "-fl",
        "--field_limit",
        default=20,
        help="The max number of field values to display per field",
    )

    args = parser.parse_args()

    project_id = args.project_url.strip("/").split("/")[6]
    api_url = (
        "/".join(args.project_url.strip("/").split("/")[:3]).strip("/") + "/api/v5"
    )
    proj_apiv5 = "{}/projects/{}".format(api_url, project_id)

    field_name_list = None
    if args.field_names:
        field_name_list = args.field_names.split("|")

    client = LuminosoClient.connect(url=proj_apiv5)
    if args.list_names:
        concept_list_names = args.list_names.split("|")
        concept_lists_raw = client.get("concept_lists/")
        concept_lists = [
            cl for cl in concept_lists_raw if cl["name"] in concept_list_names
        ]
        if not concept_lists:
            print(
                "Error: must specify at least one saved concept list. Lists available: {}".format(
                    [c["name"] for c in concept_lists_raw]
                )
            )
            concept_lists = None
    else:
        concept_lists = client.get("concept_lists/")
        if len(concept_lists) < 1:
            print("Error: project must have at least one shared concept list.")
            concept_lists = None

    if concept_lists:
        print("field_names: {}".format(field_name_list))
        print("concept_lists: {}".format([c["name"] for c in concept_lists]))
        print("field_limit: {}".format(args.field_limit))

        doc_count_by_concept_lists(
            client,
            concept_lists,
            field_names=field_name_list,
            field_limit=int(args.field_limit),
        )

        # make the output data
        odata = []
        for saved_concept in concept_lists:
            for md in saved_concept["metadata"]:
                if "values" in md:
                    for v in md["values"]:
                        if "concept_doc_counts" in v:
                            for c, count in v["concept_doc_counts"].items():
                                if count > 0:
                                    fv_pct = count / v["count"]
                                else:
                                    fv_pct = 0
                                odata.append(
                                    {
                                        "list": saved_concept["name"],
                                        "field": md["name"],
                                        "value": v["value"],
                                        "concept": c,
                                        "count": count,
                                        "field_value_percent": fv_pct,
                                    }
                                )
                            odata.append(
                                {
                                    "list": saved_concept["name"],
                                    "field": md["name"],
                                    "value": v["value"],
                                    "concept": "_ALL_",
                                    "count": v["count"],
                                    "field_value_percent": 100.0,
                                }
                            )

        if odata:
            fields = odata[0].keys()
            with open(args.output_filename, "w") as f:
                writer = csv.DictWriter(f, fields)
                writer.writeheader()
                writer.writerows(odata)

            print("Done. Data written to: {}".format(args.output_filename))
        else:
            print("Error. No data in metadata fields or shared concepts.")


if __name__ == "__main__":
    main()
