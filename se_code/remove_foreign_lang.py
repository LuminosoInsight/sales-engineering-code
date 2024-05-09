import argparse
from lingua import Language, LanguageDetectorBuilder
from luminoso_api import V5LuminosoClient as LuminosoClient
import time

languages = [Language.ENGLISH,
             Language.INDONESIAN,
             Language.CHINESE,
             Language.DUTCH,
             Language.FRENCH,
             Language.GERMAN,
             Language.ITALIAN,
             Language.JAPANESE,
             Language.KOREAN,
             Language.POLISH,
             Language.PORTUGUESE,
             Language.RUSSIAN,
             Language.SPANISH,
             Language.SWEDISH]


detector = LanguageDetectorBuilder.from_languages(*languages).build()


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def delete_docs(client, ids):
    bad_batch = batch(ids, 600)
    for bad_ids in bad_batch:
        client.post('docs/delete', doc_ids=bad_ids)
        time.sleep(1)


def get_all_docs(client, batch_size=20000):
    docs = []
    offset = 0
    while True:
        newdocs = client.get('docs', offset=offset, limit=batch_size)
        if not newdocs['result']:
            return docs
        docs.extend(newdocs['result'])
        offset += batch_size


def remove_foreign_lang(client, lang_code, threshold=.4, test=False):

    docs = get_all_docs(client)

    bad_doc_ids = []

    for doc in docs:
        confidence_values = detector.compute_language_confidence_values(doc['text'])

        if confidence_values[0].language and not confidence_values[0].language.iso_code_639_1.name.upper() in lang_code.upper():
            # print(f"{confidence_values[0].language.iso_code_639_1.name.upper()} - {confidence_values[0].value:2f}")
            print(f"{confidence_values[0].language.iso_code_639_1.name.upper()}:{confidence_values[0].value:2f}: {doc['text']}")
            bad_doc_ids.append(doc['doc_id'])

    if not test:
        delete_docs(client, bad_doc_ids)
        client.post('build')
        client.wait_for_build()
        print('{} documents not identified as "{}" removed from project.'.format(len(bad_doc_ids), lang_code))
    else:
        print('TEST ONLY: {} documents not identified as "{}" removed from project.'.format(len(bad_doc_ids), lang_code))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_url', help="The URL of the project to analyze")
    parser.add_argument('lang_code', default='en', help="The 2 character language code to retain ex. en, fr")
    parser.add_argument('--threshold', default=.4, type=float, help="Minimum threshold for desired language (ex .95 for 95%%)")
    parser.add_argument('-t', '--test', default=False,
                        action='store_true',
                        help="Only run as a test and print lists of documents that would have been deleted")
    args = parser.parse_args()

    api_url = args.project_url.split('/app')[0]
    project_id = args.project_url.strip('/ ').split('/')[-1]
    client = LuminosoClient.connect(url='{}/api/v5/projects/{}/'.format(api_url, project_id), user_agent_suffix='se_code:remove_foreign_lang')
    remove_foreign_lang(client, args.lang_code, args.threshold, args.test)
    
if __name__ == '__main__':
    main()