import argparse
from argparse import RawTextHelpFormatter
from luminoso_api import V5LuminosoClient as LuminosoClient

from se_code.copy_shared_concepts import copy_shared_concepts
from se_code.copy_shared_views import copy_shared_views

'''
Add Sentiment As Driver
Take any project and add sentiment metadata to it that will behave like a rating score. 
This script will create a new project which has everything in the original 
project (project_url) and four new metadata values on every document that has sentiment.

Positive Sentiment: The % of the terms in a document with positive sentiment, corrected for confidence.
Negative Sentiment: The % of the terms in a document with negative sentiment, corrected for confidence.
Net Sentiment: Positive Sentiment minus Negative Sentiment out of the total terms, including Neutrals. 
               Provides the net balance of sentiment in the document. Neutrality brings net sentiment lower.
Sentiment Polarization: Proportion of sentiment being pulled to Positive and Negative out of the total
                        terms, including Neutrals. Provides the level of sentiment disagreement in the 
                        document. Neutrality brings polarization lower.

These are probably most useful in the score drivers view.
'''

def split_url(project_url):
    workspace_id = project_url.strip('/').split('/')[5]
    project_id = project_url.strip('/').split('/')[6]
    api_url = '/'.join(project_url.strip('/').split('/')[:3]).strip('/') + '/api/v5'
    proj_api = '{}/projects/{}'.format(api_url, project_id)

    return(workspace_id, project_id, api_url, proj_api)


def read_documents(client, max_docs=0):
    docs = []
    while True:
        result = client.get( 
            "/docs", 
            limit=5000, 
            offset=len(docs),
            include_sentiment_on_concepts=True
        )
        if result["result"]:
            docs.extend(result["result"])
        else:
            break
        if 0 < max_docs <= len(docs):
            break
    return docs


def calcSentimentScores(positive, negative):
    multiplier = positive + negative
    numerator = min(positive, negative)
    denominator = max(positive, negative)
    if denominator <= 0:
        denominator = 1
    return [multiplier*(positive-negative), multiplier*numerator/denominator]


def write_documents(client, docs):
    offset = 0
    while offset < len(docs):
        end = min(len(docs), offset+1000)
        result = client.post('upload/', docs=docs[offset:end])
        offset = end


def main():

    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description='''Add Sentiment As Driver

  Take any project and add sentiment metadata to it that will behave like a
  rating score. This script will create a new project which has everything in
  the original project (project_url) and four new metadata values on every
  document that has sentiment.

  Positive Sentiment: The % of the terms in a document with positive
      sentiment, corrected for confidence.
  Negative Sentiment: The % of the terms in a document with negative
      sentiment, corrected for confidence.
  Net Sentiment: Positive Sentiment minus Negative Sentiment out of the total
      terms, including Neutrals. Provides the net balance of sentiment in the
      document. Neutrality brings net sentiment lower.
  Sentiment Polarization: Proportion of sentiment being pulled to Positive
      and Negative out of the total terms, including Neutrals. Provides the
      level of sentiment disagreement in the document. Neutrality brings 
      polarization lower.
  These are probably most useful in the score drivers view.
            '''
    )
    parser.add_argument('project_url',
                        help="The complete URL of the Daylight project")

    parser.add_argument('-n', '--new_project_name', default=None, required=False,
                        help='Name for the new project. Default adds ++ to the project name')
    parser.add_argument('--copy_concepts', default=True, required=False,
                        help='Copy the shared concepts and views to the new project')
    args = parser.parse_args()

    url_parts = split_url(args.project_url)

    workspace_id = url_parts[0]
    project_id = url_parts[1]
    api_url = url_parts[2]
    proj_apiv5 = url_parts[3]
    client_root = LuminosoClient.connect(api_url)
    client = LuminosoClient.connect(proj_apiv5)
    project_info = client.get()

    if args.new_project_name:
        new_project_name = args.new_project_name
    else:
        new_project_name = project_info['name']+"++"

    print("Reading all documents")
    docs = read_documents(client)

    print("Calculating sentiment as a driver")
    new_docs = []
    for d in docs:
        sentiment_count_neutral = 0
        sentiment_count_positive = 0
        sentiment_count_negative = 0
        sentiment_negative_total = 0
        sentiment_positive_total = 0
        sentiment_neutral_total = 0
        for t in d['terms']:
            if t['sentiment'] == 'negative':
                sentiment_count_negative += 1
                sentiment_negative_total += t['sentiment_confidence']
            elif t['sentiment'] == 'positive':
                sentiment_count_positive += 1
                sentiment_positive_total += t['sentiment_confidence']
            elif t['sentiment'] == 'neutral':
                sentiment_count_neutral += 1
                sentiment_neutral_total += t['sentiment_confidence']

        nd = {'text': d['text'],
              'title': d['title'],
              'metadata': d['metadata']}

        total_confidence = sentiment_negative_total + sentiment_positive_total + sentiment_neutral_total
        if total_confidence > 0:

            avg_conf_neg = sentiment_negative_total / total_confidence
            avg_conf_pos = sentiment_positive_total / total_confidence

            s_conf = calcSentimentScores(avg_conf_pos, avg_conf_neg)

            nd['metadata'].append({'name': 'Net Sentiment', 'type': 'number', 'value': s_conf[0]})
            nd['metadata'].append({'name': 'Sentiment Polarization', 'type': 'number', 'value': s_conf[1]})
            nd['metadata'].append({'name': 'Negative Sentiment', 'type': 'number', 'value': avg_conf_neg})
            nd['metadata'].append({'name': 'Positive Sentiment', 'type': 'number', 'value': avg_conf_pos})

        new_docs.append(nd)

    # create the new project
    print("Creating new project: "+new_project_name)
    new_prj_info = client_root.post("/projects", name=new_project_name,
                                    language=project_info['language'], 
                                    workspace_id=workspace_id)
    new_client = client_root.client_for_path('/projects/'+new_prj_info['project_id'])
    write_documents(new_client, new_docs)

    print("Copying shared views and concept lists")
    copy_shared_concepts(client, new_client)
    copy_shared_views(client, new_client)

    print("Waiting for new project build...")
    new_client.post("build")
    new_client.wait_for_build()


if __name__ == '__main__':
    main()
