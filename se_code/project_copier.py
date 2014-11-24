import argparse
import logging
from luminoso_api import LuminosoClient, LuminosoAuthError

def copy_project(project_path, username, destination=None, owner=None,
                 deployed=False):
    """
    Required parameters:
        project_path - the account ID of the project to copy from, an
            underscore, and the five-character project ID of the project to
            copy from.
        username - a Luminoso username that has permissions on the appropriate
            accounts and projects.

    Optional parameters:
        destination - the name of the copy, defaults to "Copy of <project>"
        owner - the account ID of the account that will own the copy. Defaults
            to the original account.
        deployed - A boolean value indicating whether these projects are on
            the deployed version of the Luminoso system. If false, it connects
            to the staged version. Defaults to true.

    A wrapper around the POST 'copy' endpoint, so that it can be used from the
    command line.
    """

    project_path = project_path.replace('_', '/')

    if deployed:
        client = LuminosoClient.connect('projects', username=username)
    else:
        client = LuminosoClient.connect('http://api.staging.lumi/v4/projects',
                                        username=username)

    project = client.change_path(project_path)
    
    try:
        name = project.get()['name']
    except LuminosoAuthError:
        raise RuntimeError('Luminoso authorization error on project '  +
                           project_path + '. Possibly it does not exist.')

    # I'm sure there's a better way to do this... Maybe with **kwargs?
    if destination is None:
        destination = 'Copy of ' + name
    if owner is None:
        owner = project_path.partition('/')[0]

    project.post('copy', destination=destination, owner=owner)
    # What's the best way to change this to a logging format?
    print('Copied', name, 'to account', owner)


def main():
    description = 'Make a copy of a Luminoso project.'
    project_path_help = 'The account ID of the owner of the \
                         project to copy from, an underscore, and the five- \
                         character project ID of the project to copy from.'
    username_help = 'A Luminoso username that has permissions on the \
                     appropriate accounts and projects'
    destination_help = 'The name of the copy, defaults to "Copy of <project>"'
    owner_help = 'the account ID of the account that will own the copy. \
                  Defaults to the original account.'
    deployed_help = 'A boolean value indicating whether these projects are on \
                     the deployed version of the Luminoso system. If false, \
                     it connects to the staged version. Defaults to false.'

    logging.basicConfig()
    log = logging.getLogger('topic-copier')

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('project_path', help=project_path_help)
    parser.add_argument('username', help=username_help)
    parser.add_argument('--destination', help=destination_help)
    parser.add_argument('--owner', help=owner_help)
    parser.add_argument('-d', '--deployed', help=deployed_help,
                        action='store_true')

    args = parser.parse_args()

    try:
        copy_project(project_path=args.project_path,
                     username=args.username,
                     destination=args.destination,
                     owner=args.owner,
                     deployed=args.deployed)
    except RuntimeError as e:
        log.error('LuminosoAuthError:' + str(e))
    except Exception as e:
        log.error('This program hit an exception (%s: %s).',
                  e.__class__.__name__, e)

if __name__ == '__main__':
    main()
