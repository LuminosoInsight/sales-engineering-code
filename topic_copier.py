import argparse
import logging
from operator import itemgetter
from luminoso_api import LuminosoClient, LuminosoAuthError
import luminoso_api


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
    log.info('Topic posted: %s', topic)


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


if __name__ == '__main__':
    # Is there a way to grab this information straight from the docstring?
    description = 'Copy topics from one Luminoso project to another.'
    old_project_path_help = 'The eight-character account ID of the project \
                             to copy from, an underscore, and the \
                             five-character account ID of the project to copy \
                             from.'
    new_project_path_help = 'The eight-character account ID of the project to \
                             copy to, an underscore, and the five-character \
                             account ID of the project to copy to.'
    username_help = 'A Luminoso username that has permissions on the \
                     appropriate accounts and projects'
    deployed_help = 'A boolean value indicating whether these projects are on \
                     the deployed version of the Luminoso system. If false, \
                     it connects to the staged version. Defaults to false.'
    sort_help = 'A boolean value indicating whether the topics should be \
                 sorted by color before posting. Defaults to false.'

    logging.basicConfig()
    log = logging.getLogger('topic-copier')

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('old_project_path', help=old_project_path_help)
    parser.add_argument('new_project_path', help=new_project_path_help)
    parser.add_argument('username', help=username_help)
    parser.add_argument('-d', '--deployed', help=deployed_help,
                        action='store_true')
    parser.add_argument('-s', '--sort', help=sort_help, action='store_true')
    args = parser.parse_args()
    try:
        topic_copier(old_project_path=args.old_project_path,
                     new_project_path=args.new_project_path,
                     username=args.username,
                     deployed=args.deployed,
                     sort=args.sort)
    except RuntimeError as e:
        log.error('LuminosoAuthError:' + str(e))
    except Exception as e:
        log.error('This program hit an exception (%s: %s).',
                  e.__class__.__name__, e)
