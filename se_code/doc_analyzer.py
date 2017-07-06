import click
import json

from luminoso_api import LuminosoClient
from luminoso_api.json_stream import stream_json_lines
from luminoso_api.upload import batches



def sample_iterator(iterator, sample_rate=10):
    """
    Yield one out of every `sample_rate` items from an iterator.
    """
    for i, item in enumerate(iterator):
        if i % sample_rate == 0:
            yield item


def create_sampled_project(account_client, project_name, infile, language=None, sample_rate=10):
    info = account_client.post(name=project_name)
    project_id = info['project_id']
    print("Created project %r with ID %s" % (project_name, project_id))

    project = account_client.change_path(project_id)
    docs = sample_iterator(stream_json_lines(infile), sample_rate)

    for i, batch in enumerate(batches(docs, 100)):
        project.upload('docs', list(batch))
        print('Uploaded batch #%d' % i)

    print('Calculating.')
    kwargs = {}
    if language is not None:
        kwargs['language'] = language
    job_id = project.post('docs/recalculate', **kwargs)
    project.wait_for(job_id)

    # Return to the start of the file with our input documents
    infile.seek(0)
    return project


def process_docs(project, infile, outfile):
    """
    Upload documents from `infile` to an existing project, getting back their analyzed
    form and writing them to `outfile`.
    """
    print("Analyzing documents.")
    n_analyzed = 0
    for batch in batches(stream_json_lines(infile), 100):
        batch = list(batch)
        vectorized = project.upload('docs/vectors', batch)
        for doc in vectorized:
            print(json.dumps(doc, ensure_ascii=False), file=outfile)
        n_analyzed += len(batch)
        print("Analyzed %d documents" % n_analyzed)


def sample_and_vectorize(account_client, project_name, infile, outfile, language, sample_rate=10):
    """
    Read a JSON stream of documents from `infile`, and write the analyzed
    versions of them to `outfile`.

    They will be analyzed according to the Luminoso project named `project_name`,
    which will be created if it doesn't exist. If the project needs to be created,
    it will be created from a 1/N sample of the documents, where N = `sample_rate`.
    """
    projects = account_client.get(name=project_name)
    project_id = None
    if projects:
        if len(projects) > 1:
            raise ValueError("More than one project is named %r. That's not supposed to happen." % project_name)
        info = projects[0]
        if info['current_assoc_version'] == -1:
            print("Deleting broken project.")
            account_client.delete(info['project_id'])
        else:
            project_id = info['project_id']

    if project_id is None:
        project = create_sampled_project(account_client, project_name, infile, language, sample_rate)
    else:
        project = account_client.change_path(project_id)

    process_docs(project, infile, outfile)


@click.command()
@click.argument('account_id')
@click.argument('project_name')
@click.argument('username')
@click.argument('infile', type=click.File('r', encoding='utf-8'))
@click.argument('outfile', type=click.File('w', encoding='utf-8'))
@click.option('--api-url', '-a', default='https://analytics.luminoso.com/api/v4')
@click.option('--language', '-l', default=None)
@click.option('--sample-rate', '-s', type=int, default=10)
def main(account_id, project_name, username, infile, outfile, api_url, language, sample_rate):
    account_url = '%s/projects/%s' % (api_url, account_id)
    account_client = LuminosoClient.connect(account_url, username=username)
    sample_and_vectorize(account_client, project_name, infile, outfile, language, sample_rate)


if __name__ == '__main__':
    main()
