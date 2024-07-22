import argparse
from argparse import RawTextHelpFormatter
from luminoso_api import V5LuminosoClient as LuminosoClient

from se_code.copy_shared_concepts import copy_shared_concepts
from se_code.copy_shared_views import copy_shared_views

import openai
import os
import time

'''
Use ChatGPT to add Plutchik emotions to a dataset

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


# Initialize the OpenAI API with your key
openai.api_key = os.environ['OPENAI_API_KEY']

plutchik_lists = {
    'Emotion 1': ["A:Ecstasy", "B:Admiration", "C:Terror", "D:Amazement", "E:Grief", "F:Loathing", "G:Rage", "H:Vigilance"],
    'Emotion 2': ["A:Joy", "B:Trust", "C:Fear", "D:Surprise","E:Sadness", "F:Disgust", "G:Anger", "H:Anticipation"],
    'Emotion 3': ["A:Serenity", "B:Acceptance", "C:Apprehension", "D:Distraction", "E:Pensiveness", "F:Boredom", "G:Annoyance", "H:Interest"],
    'Emotion 4': ["Love", "Submission", "Awe", "Disapproval", "Remorse", "Contempt", "Aggressiveness", "Optimism"]
}


def place_emotion(emotion):
    for k, l in plutchik_lists.items():
        l2 = [e for e in l if emotion.lower() in e.lower()]
        if len(l2) == 1:
            return (k, l2[0])
    return None


def identify_emotions(text):

    #shortened_text = text.split(" ")
    #shortened_text = " ".join(shortened_text[0:100])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with product reviews, and your task is to list the plutchik emotions present and provide them as a comma delimited list. Only include emotions you are highly confident about."
            },
            {
                "role": "user",
                "content": " ".join(text.split(" ")[0:100])
            }
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Use the OpenAI API to get the model's response
    #response = openai.Completion.create(
    #    engine="davinci", 
    #    prompt=f"Give me a comma separated list the emotions from Plutchik's Wheel of Emotions present in the following text: '{text}'",
    #    max_tokens=100  # Adjust based on your needs
    #)

    # Extract the text from the model's response
    emotions = response.choices[0].message.content.strip().split(', ')

    return emotions



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
        description='''Add Plutchik emotions as metadata to each document


            '''
    )
    parser.add_argument('project_url',
                        help="The complete URL of the Daylight project")

    parser.add_argument('-n', '--new_project_name', default=None, required=False,
                        help='Name for the new project. Default adds ++emotions to the project name')
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
        new_project_name = project_info['name']+"+Plutchik"

    print("Reading all documents")
    docs = read_documents(client)

    print("Calculating plutchik emotions as metadata")
    new_docs = []
    for d in docs:

        done = False
        retry_count = 0

        while not done or retry_count>10:
            try:
                emotions = identify_emotions(d['text'])
                done = True
            # except openai.error.ServiceUnavailableError:
            #     print(f"server busy. Len {len(new_docs)}. Sleeping then retrying")
            #     time.sleep(1)
            #     retry_count += 1
            # except openai.error.Timeout:
            #     print(f"server timed out. Len {len(new_docs)}. Sleeping then retrying")
            #     time.sleep(1)
            #     retry_count += 1
            except Exception as e:

                print(f"General error {e}. Len {len(new_docs)}. Sleeping then retrying")
                time.sleep(1)
                retry_count += 1

        if done:
            nd = {'text': d['text'],
                  'title': d['title'],
                  'metadata': d['metadata']}

            for e in emotions:
                e_tup = place_emotion(e)
                if e_tup is not None:
                    nd['metadata'].append({'name': e_tup[0], 'type': 'string', 'value': e_tup[1]})
                #else:
                #    print(f"emotion not in list: {e}")
            new_docs.append(nd)
        else:
            print(f"skipped document: {d['text'][0:100]}")

        if (len(new_docs) % 100) == 0:
            print(f"completed: {len(new_docs)} of {len(docs)}")

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
