import logging
from operator import itemgetter
from luminoso_api import LuminosoClient, LuminosoAuthError

LOG = None

# Copied (& modified) from lumi_auth/cli, because it seems silly to have a
# dependency just for this.
def bool_prompt_with_default(prompt):
    """
    Given a string to prompt the user with, prompt them until they give an
    appropriate answer: return True if the answer starts with Y or y, and False
    if it starts with N or n.
    """
    while True:
        res = input(prompt).strip().lower()
        if res.startswith('y') or res == '':
            return True
        if res.startswith('n'):
            return False


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


def topic_copier(old_account, old_project, new_account, new_project, username,
                 deployed=False, sort=False):
    """
    Required parameters:
        old_account - the eight-character account ID of the project to copy from
        old_project - the five-character project ID of the project to copy from
        new_account - the eight-character account ID of the project to copy to
        new_project - the five-character project ID of the project to copy to
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

    if deployed:
        client = LuminosoClient.connect('projects', username=username)
    else:
        client = LuminosoClient.connect('http://api.staging.lumi/v4/projects',
                                        username=username)

    old_project_path = '%s/%s' % (old_account, old_project)
    new_project_path = '%s/%s' % (new_account, new_project)
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

    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger('topic-copier')

    print('\nCopy topics from one Luminoso project to another.')
    old_account = input('Account ID of project to copy topics FROM: ')
    old_project = input('Project ID of project to copy topics FROM: ')
    new_account = input('Account ID of project to copy topics TO '
                        '(Leave blank if same account): ') or old_account
    new_project = input('Project ID of project to copy topics TO: ')
    username = input('Luminoso username: ')
    deployed = bool_prompt_with_default('Use deployed system '
                                        '(as opposed to staging) [Y/n]? ')
    sort = not bool_prompt_with_default('Use same topic order as original '
                                       '(as opposed to sorting by color) '
                                       '[Y/n]? ')

    try:
        topic_copier(old_account, old_project, new_account, new_project,
                     username, deployed=deployed, sort=sort)
    except RuntimeError as e:
        LOG.error('LuminosoAuthError:' + str(e))
    except Exception as e:
        LOG.error('This program hit an exception (%s: %s).',
                  e.__class__.__name__, e)

if __name__ == '__main__':
    main()
