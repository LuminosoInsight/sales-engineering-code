import argparse
import logging
from operator import itemgetter
from luminoso_api import LuminosoClient, LuminosoAuthError

LOG = None

def post_topic(project, topic):
    """
    Required parameters:
        project - a LuminosoClient pointed at a particular project_id
        topic - a topic dictionary, presumably retrieved from GET /topics/

    Result: the topic is posted to the new project
        (and prints it to the screen).

    Formats a topic dictionary from Luminoso into an appropriate format for
    reupload by deleting the unnecessary arguments, and then POSTs it to the
    appropriate project.
    """

    del topic['vector']
    del topic['_id']
    project.post('topics', **topic)
    if LOG:
        LOG.info('Topic posted: %s', topic)


def topic_copier(old_project_path, new_project_path, username,
                 deployed=False, sort=False):
    """
    Required parameters:
        old_project_path - the eight-character account ID of the project to
            copy from, an underscore, and the five-character project ID of the
            project to copy from.
        new_project_path - the eight-character account ID of the project to
            copy to, an underscore, and the five-character project ID of the
            project to copy to.
        username - a Luminoso username that has permissions on the appropriate
            accounts and projects.

    Optional parameters:
        deployed - A boolean value indicating whether these projects are on
            the deployed version of the Luminoso system. If false, it connects
            to the staged version. Defaults to true.
        sort - A boolean value indicating whether the topics should be sorted
            by color before posting. Defaults to false, in which case, topic
            order is preserved.

    Result: all topics are copied from one project to another.
    """

    # Ensure that the paths are correctly forwarded
    old_project_path = old_project_path.replace('_', '/')
    new_project_path = new_project_path.replace('_', '/')

    if deployed:
        client = LuminosoClient.connect('projects', username=username)
    else:
        client = LuminosoClient.connect('http://api.staging.lumi/v4/projects',
                                        username=username)

    old_project = client.change_path(old_project_path)
    new_project = client.change_path(new_project_path)

    # Test to ensure the paths are invalid
    try:
        old_project.get()
    except LuminosoAuthError:
        raise RuntimeError('Luminoso authorization error on project '  +
                           old_project_path + '. Possibly it does not exist.')
    try:
        new_project.get()
    except LuminosoAuthError:
        raise RuntimeError('Luminoso authorization error on project '  +
                           new_project_path + '. Possibly it does not exist.')

    topics = old_project.get('topics')

    if sort:
        topics.sort(key=itemgetter('color'))
    # Topics are reversed so they're posted in the correct order.
    for topic in reversed(topics):
        post_topic(new_project, topic)


def main():
    global LOG

    description = 'Copy topics from one Luminoso project to another.'
    deployed_help = 'A boolean value indicating whether these projects are on \
                     the deployed version of the Luminoso system. If false, \
                     it connects to the staged version. Defaults to false.'
    sort_help = 'A boolean value indicating whether the topics should be \
                 sorted by color before posting. Defaults to false.'

    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger('topic-copier')

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--deployed', help=deployed_help,
                        action='store_true')
    parser.add_argument('-s', '--sort', help=sort_help, action='store_true')
    args = parser.parse_args()

    print(description + '\n')
    old_account = input('Account ID of project to copy topics FROM: ')
    old_project = input('Project ID of project to copy topics FROM: ')
    if input('Copy to a project in the same account? (y/n): ').lower() == 'n':
        new_account = input('Account ID of project to copy topics TO: ')
    else:
        new_account = old_account
    new_project = input('Project ID of project to copy topics TO: ')
    old_project_path = '%s_%s' % (old_account, old_project)
    new_project_path = '%s_%s' % (new_account, new_project)
    username = input('Luminoso username: ')

    try:
        topic_copier(old_project_path=old_project_path,
                     new_project_path=new_project_path,
                     username=username,
                     deployed=args.deployed,
                     sort=args.sort)
    except RuntimeError as e:
        LOG.error('LuminosoAuthError:' + str(e))
    except Exception as e:
        LOG.error('This program hit an exception (%s: %s).',
                  e.__class__.__name__, e)

if __name__ == '__main__':
    main()
