from luminoso_api import V5LuminosoClient as LuminosoClient
import collections, csv, json, datetime, time, argparse


def get_all_docs(client):
    docs = []
    while True:
        new_docs = client.get(
            "docs", include_sentiment=True, limit=25000, offset=len(docs)
        )
        if new_docs["result"]:
            docs.extend(new_docs["result"])
        else:
            return docs


def search_all_doc_ids(client, concepts, match_type="both"):
    docs = []
    while True:
        if "both" in match_type:
            new_docs = client.get(
                "docs",
                search={"texts": concepts},
                fields=["doc_id", "match_score"],
                limit=25000,
                offset=len(docs),
            )
        else:
            new_docs = client.get(
                "docs",
                search={"texts": concepts},
                fields=["doc_id", "match_score"],
                limit=25000,
                offset=len(docs),
                match_type=match_type,
            )
        if new_docs["result"]:
            docs.extend(
                {"doc_id": d["doc_id"], "match_score": d["match_score"]}
                for d in new_docs["result"]
            )
        else:
            return docs


def add_relations(
    client,
    docs,
    add_concept_relations=False,
    add_concept_list=False,
    match_type="both",
    add_match_score=False,
    add_saved_concept_sentiment=False,
):
    # get the list of saved concepts with included sentiment
    saved_concepts = client.get(
        "concepts/sentiment", concept_selector={"type": "saved"}
    )["match_counts"]

    # pre-calculate the top sentiment share
    for sc in saved_concepts:
        sc["sentiment"] = collections.Counter(sc["sentiment_share"]).most_common()[0][0]

        search_results = search_all_doc_ids(client, sc["texts"], match_type=match_type)
        sc["match_scores_by_id"] = {
            c["doc_id"]: c["match_score"] for c in search_results
        }

    # filter out metadata that matches our current saved concepts
    clist = [sc["name"] for sc in saved_concepts]
    for d in docs:
        d["metadata"] = [md for md in d["metadata"] if md["name"] not in clist]

    # loop through each doc and list of saved concepts finding if the doc is in that search result
    # add metadata for its yes/no relation to each saved concept
    # if it is not associated with any saved concept, mark it as an outlier
    for d in docs:
        in_none = True
        doc_concept_list = []
        for sc in saved_concepts:
            if d["doc_id"] in sc["match_scores_by_id"]:
                in_none = False
                if add_concept_relations:
                    d["metadata"].append(
                        {"name": sc["name"], "type": "string", "value": "yes"}
                    )
                    if add_match_score:
                        d["metadata"].append(
                            {
                                "name": sc["name"] + " match_score",
                                "type": "number",
                                "value": sc["match_scores_by_id"][d["doc_id"]],
                            }
                        )
                doc_concept_list.append(sc["name"])

                if add_saved_concept_sentiment:
                    d["metadata"].append(
                        {
                            "name": sc["name"] + " sentiment",
                            "type": "string",
                            "value": sc["sentiment"],
                        }
                    )
            else:
                if add_concept_relations:
                    d["metadata"].append(
                        {"name": sc["name"], "type": "string", "value": "no"}
                    )
                    if add_match_score:
                        d["metadata"].append(
                            {
                                "name": sc["name"] + " match_score",
                                "type": "number",
                                "value": 0,
                            }
                        )

        if add_concept_relations:
            if in_none:
                d["metadata"].append(
                    {"name": "doc_outlier", "type": "string", "value": "yes"}
                )
                doc_concept_list.append("outlier")
            else:
                d["metadata"].append(
                    {"name": "doc_outlier", "type": "string", "value": "no"}
                )
        if add_concept_list:
            d["metadata"].append(
                {
                    "name": "concept_list",
                    "type": "string",
                    "value": "|".join(doc_concept_list),
                }
            )


# flatten the doc structure for csv export
def flatten_docs(docs, date_format):

    # dict for sorting the names once the values have been changed to have lumi types
    field_name_dict = {
        fn: fn for fn in [md["name"] for d in docs for md in d["metadata"]]
    }

    flat_docs = []
    for d in docs:
        flat_doc = {"text": d["text"], "title": d["title"]}

        for md in d["metadata"]:
            # add the type to the field name for export
            md_name = "%s_%s" % (md["type"], md["name"])

            # save the luminoso name for this for later sorting
            field_name_dict[md["name"]] = md_name

            if md["type"] == "date":
                try:
                    flat_doc[md_name] = datetime.datetime.fromtimestamp(
                        int(md["value"])
                    ).strftime(date_format)
                except ValueError:
                    flat_doc[md_name] = "%s" % md["value"]
            else:
                flat_doc[md_name] = md["value"]

        flat_docs.append(flat_doc)

        field_names = ["text", "title"]
        field_names.extend(
            {
                k: v
                for k, v in sorted(field_name_dict.items(), key=lambda item: item[0])
            }.values()
        )

    return field_names, flat_docs


def write_to_csv(filename, docs, field_names, encoding="utf-8"):
    with open(filename, "w", encoding=encoding) as f:
        writer = csv.DictWriter(f, field_names)
        writer.writeheader()
        writer.writerows(docs)
    print("Wrote %d docs to %s" % (len(docs), filename))


def main():
    parser = argparse.ArgumentParser(
        description="Download documents from an Analytics project and write to CSV."
    )
    parser.add_argument("project_url", help="The URL of the project to analyze")
    parser.add_argument(
        "filename", help="Name of CSV file to write project documents to"
    )
    parser.add_argument(
        "-e",
        "--encoding",
        default="utf-8",
        help="Encoding type of the file to write to",
    )
    parser.add_argument(
        "-d", "--date_format", default="%Y-%m-%d", help="Format of timestamp"
    )

    parser.add_argument(
        "-c",
        "--concept_relations",
        default=False,
        action="store_true",
        help="Add columns for saved concept relations and outliers",
    )
    parser.add_argument(
        "-l",
        "--concept_list",
        default=False,
        action="store_true",
        help="Add columns for saved concept relations and outliers",
    )
    parser.add_argument(
        "-mt",
        "--match_type",
        default=None,
        help="For concept relations use exact, conceptual or both when searching",
    )
    parser.add_argument(
        "-ms",
        "--match_score",
        default=False,
        action="store_true",
        help="For concept relations also include the match_score",
    )
    parser.add_argument(
        "-s",
        "--concept_relations_sentiment",
        default=False,
        action="store_true",
        help="Add saved concept sentiment to the concept_relations",
    )

    args = parser.parse_args()

    # require that -c is set if using any of the other flags associated with concept relations
    if (
        args.concept_list
        or args.match_score
        or args.concept_relations_sentiment
        or args.match_type is not None
    ) and (not args.concept_relations):
        parser.error("-l, -mt, -ms, -s all require -c")

    # set the default value for match type. Not using this in add_argument, because
    # need to know if user set for checking that -c is also set
    if not args.match_type:
        args.match_type = "both"

    api_url = args.project_url.split("/app")[0]
    project_id = args.project_url.strip("/ ").split("/")[-1]

    account_id = args.project_url.strip("/").split("/")[5]
    project_id = args.project_url.strip("/").split("/")[6]
    api_url = (
        "/".join(args.project_url.strip("/").split("/")[:3]).strip("/") + "/api/v5"
    )
    proj_apiv5 = "{}/projects/{}".format(api_url, project_id)

    client = LuminosoClient.connect(url=proj_apiv5)

    docs = get_all_docs(client)
    if args.concept_relations or args.concept_list:
        add_relations(
            client,
            docs,
            args.concept_relations,
            args.concept_list,
            match_type=args.match_type,
            add_match_score=args.match_score,
            add_saved_concept_sentiment=args.concept_relations_sentiment,
        )

    field_names, docs = flatten_docs(docs, args.date_format)

    write_to_csv(args.filename, docs, field_names, encoding=args.encoding)


if __name__ == "__main__":
    main()
