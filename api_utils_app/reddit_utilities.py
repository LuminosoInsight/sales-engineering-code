import praw, re, csv, argparse

from ftfy import fix_text

THREESHORTENER_RE = re.compile(r'(\D+?)\1{3,}')

SEARCH_TYPES = ['top', 'controversial', 'new']
SEARCH_PERIODS = ['week', 'hour', 'day', 'month', 'year', 'all']
USERNAME = ''
PASSWORD = ''


def threeshorten(text):
    """
    Shortens a pattern in the text to a maximum 3 repeats of the pattern

    :param text: A string

    :return: The string with all lengthy patterns shortened to a max of 3
    repetitions
    """
    def _subst(match):
        return match.group(1) * 3

    return re.sub(THREESHORTENER_RE, _subst, text)


def clean_text(text):
    return threeshorten(fix_text(text))

def get_reddit_api():
    """ Using the secrets, authenticate with the api """
    reddit = praw.Reddit(
        client_id='j3TJpj9nl_069g',
        client_secret='Nmh0dNjH67YkbyLx1McKpGvdSQk',
        user_agent='personal use script'
    )
    return reddit


def get_posts_from_past(
    reddit, subreddit_name, start_datetime, sort_type, time_frame
):
    """
    Get posts in a subreddit from recent history. Returns an iterable of
    posts. The `sort_type` is similar as reddit's sort by feature, and can be
    one of:

        ('new', 'hot', 'contreversial', 'top', 'rising', 'best')

    `time_frame` will filter posts to within that time frame, and can be one
    of:

        ('all', 'year', 'month', 'week', 'day', 'hour')

    Note: If your start_datetime is greater than your `time_frame` then posts
    will be removed.
    Note: By default this limits posts to 1000
    """
    # TODO: Remove time_frame as an arg and choose an option based on start_datetime
    # TODO: Does this need to be paginated?
    subreddit = reddit.subreddit(subreddit_name)
    if sort_type == 'top':
        submissions = subreddit.top(time_frame, limit=1000)
    elif sort_type == 'controversial':
        submissions = subreddit.controversial(time_frame, limit=1000)
    else:
        submissions = subreddit.new(limit=1000)

    posts = {}
    for s in submissions:
        if s.created >= start_datetime.timestamp():
            posts[s.id] = s.title
    return posts


def get_posts_by_name(reddit, subreddit_name, title_queries):
    """
    Get a subreddit's posts that match any of the queries in within the list of
    title queries. `title_queries` is intended to be a list of post titles,
    though it is actually a query.
    """
    posts = {}
    subreddit = reddit.subreddit(subreddit_name)
    for title_query in title_queries:
        search_posts = subreddit.search(title_query, limit=1)
        # TODO: search_posts only ever returns 1 post..
        for post in search_posts:
            posts[post.id] = post.title
    return posts


def parse_comment_text(comment):
    text = clean_text(comment.body)
    text_split = text.split(' ')
    text = []
    for t in text_split:
        if 'http' not in t and '//www.' not in t:
            text.append(t)
    text = ' '.join(text)
    return text


def create_metadata(document, meta_type, name, value):
    #document['metadata'].append({
    #    'type': meta_type,
    #    'name': name,
    #    'value': value
    #})
    document['%s_%s' % (meta_type, name)] = value


def get_docs_from_comments(posts, reddit):
    """
    Iterate over and create a document from the flattened comments for each
    post. Extracts and adds metadata to the document if it exists as well.

    Returns a generator that yields documents
    """
    # TODO: Refactor even more! (pull out the comment fetching)
    # TODO: Understand praw's comment fetching better so we can control batch
    # size there as well
    for post_id in posts:
        submission = reddit.submission(id=post_id)
        # `replace_more` loads the 'more comments' from a post into a
        # conveniently flattened list
        submission.comments.replace_more(limit=None)
        # Keep track of comment_id's within a thread. The comments retrieved
        # are ordered such that a parent is iterated before it's child. Because
        # of this, and because we store each comment's id, we can easily lookup
        # a 'nested' comment's thread by checking what thread it's parent is in.
        threads = {}
        for comment in submission.comments.list():
            doc = {}
            text = parse_comment_text(comment)
            doc['text'] = text
            doc['title'] = comment.id
            # Initial metadata setup
            #doc['metadata'] = []
            create_metadata(doc, 'date', 'Post Date', comment.created_utc)
            # Check for author
            if comment.author:
                create_metadata(
                    doc, 'string', 'Author Name', comment.author.name
                )
            else:
                create_metadata(doc, 'string', 'Author Name', '[Deleted]')

            if comment.parent_id.partition('_')[-1] == post_id:
                threads[comment.id] = set([comment.id])
                create_metadata(doc, 'string', 'Comment Type', 'parent')
                create_metadata(doc, 'string', 'Thread', comment.id)

            else:
                for thread in threads:
                    if comment.parent_id.partition('_')[-1] in threads[thread]:
                        threads[thread].add(comment.id)
                        create_metadata(doc, 'string', 'Thread', thread)
                        create_metadata(doc, 'string', 'Comment Type', 'child')
                        break
            #doc['metadata'].append({
            #    'type': 'string',
            #    'name': 'Reddit Post',
            #    'value': posts[post_id]
            #})
            doc['string_Reddit Post'] = posts[post_id]
            yield doc

def write_to_csv(filename, docs, fields):
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(docs)

def main():
    parser = argparse.ArgumentParser(
        description='Export reddit data to CSV files.'
    )
    parser.add_argument('--request_type', default="name", help="Type of reddit request default=name.  Possible values: [name,date]")
    parser.add_argument('subreddit_name', help="Enter the subreddit name")
    parser.add_argument('post_names', help="Enter the post reddit name")
    args = parser.parse_args()

    request_type = args.request_type
    subreddit_name = args.subreddit_name
    post_names = args.post_names

    fields = ['text', 'title', 'date_Post Date', 'string_Author Name', 'string_Comment Type', 'string_Thread', 'string_Reddit Post', ]
    reddit = get_reddit_api()
    posts = get_posts_by_name(reddit, subreddit_name, post_names)
    docs = get_docs_from_comments(posts, reddit)
    write_to_csv('%s docs.csv' % subreddit_name, docs, fields)

if __name__ == '__main__':
    main()
