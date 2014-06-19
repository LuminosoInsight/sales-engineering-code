Sales Engineering Code
======================

This repository holds code written by Luminosans outside the development
team; it includes code that may be sent to clients, and code used internally
for our own purposes.

Procedures
----------
Files that you put in the `se_code` directory can be run directly via your
favorite file-running procedure (ipython, for instance, has `%run <filename>`),
and can also be imported via `import se_code.<filename>`. You can also make
subdirectories of `se_code`; if you do, creating a blank `__init__.py` file in
your subdirectory will allow the same kind of importing, i.e. `import se_code.<subdirectory>.<filename>`.

However: please do not add files directly to the master branch of this
repository! Instead, there are a few Git commands that will allow you to put
things into a separate, not-yet-approved "branch", and then request that the
development team review your code.  The basic procedure, annotated with numbers
in brackets, is:

```
[1] git checkout master
[2] git checkout -b <descriptive-branch-name>
[3] <code editing goes here>
[4] git commit -a
[5] git push -u origin <descriptive-branch-name>
```

The idea is that, first, you want to make sure you're in the "master"
branch. [1] (`git status` will tell you what branch you're in, or just typing
`git checkout master` will switch you to it if you're not already there.) You
then want to create a new branch [2], something like `git checkout -b
code-for-InfoCompuCorp` or `git checkout -b topic-copying-code`. `checkout`
tells git to switch you to that branch; `-b` tells it you're making a new
branch.  Once you're in that branch, you can add or modify code to your heart's
content [3].  When you're done you "commit" your changes: the command in [4]
commits everything that's changed, or alternately you can use `git add <files>`
to add just those files you want to add, and then `git commit` to commit the
ones you've added.  Finally, push it to GitHub [5]. If you've previously put it
on GitHub and are making changes, you can just type `git push` at this step;
the command given here both pushes the code and tells GitHub about your new
branch.

When you want someone to review your code, you want to make a "pull request"
(or PR).  Go to https://github.com/LuminosoInsight/sales-engineering-code/pulls
and click "New pull request". You can click on the name of your branch, and
then click "Create pull request", which will take you to a page where you can
write a description of the PR and click the button that creates it.  After
that, someone from the Dev team will look over your code and possibly leave
comments or discuss the details with you.

Make sure that it's clear what your code *should* do! We encourage comments,
docstrings, and informative commit messages.  ("Some more code" is not an
informative commit message...)  See https://help.github.com/articles/create-a-repo and http://git-scm.com/documentation for more information on using git.