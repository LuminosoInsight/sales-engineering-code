Sales Engineering Code
======================

This repository holds code written by Luminosans outside the development
team; it includes code that may be sent to clients, and code used internally
for our own purposes. This code is considered unsupported and for demonstration purposes only.

## Downloading the repository

To start you need to download the repository
`git clone https://github.com/LuminosoInsight/sales-engineering-code/`

## Creating a python virtual environment

You will want to create a Python virtual enviroment. It isn't required, but it  will help manage dependencies between different projects or versions. Open a Terminal, navigate to the place you want your environment to live. (Note: you'll need to navigate here to start the environment, so don't bury it too deep). It's probably best to just put it in the sales-engineering-code folder

1. Create the virtual environment (Python 3):

    `python -m venv venv`

1. Run Python in your newly created environment:

    `source venv/bin/activate`

1. Test your new environment by opening Python:

    `python`

1. To log out of the environment:

    `deactivate`


## Installation for Luminoso Employees

After downloading the repository, the code package can be installed into your 
python environment with

`python setup.py develop`

## Running the web application

You can run the web application\
`cd api-utils-app`\
`python web_app.py`

Navigate your browser to (http://127.0.0.1:5000/)

##  Packaging the wheel file for customers

The se_code can be packaged for sharing with customers by using a wheel file.

`cd sales-engineering-code`
`python setup.py sdist bdist_wheel`

This will create a folder called dist with two files in it.

> se_code-0.2-py3-none-any.whl\
> se_code-0.2.tar.gz

You can give the .whl file to a customer and it contains all the scripts they may need. This code is considered demonstration code and not supported. The tar.gz file should be included if the customer would like to extend or modify the functionality of these scripts that call the Luminoso API.

## Customer Installation

Once you have the .whl file you can run the following commands to setup the Luminoso sales engineering example scripts.

`pip install dist/se_code-0.2-py3-none-any.whl`

Once installed, the wheel included shortcuts to run any of the scripts without the python interpreter on the command line. For instance the customer can simply run.

`lumi_doc_downloader --help`

A customer installation document is available [here](https://docs.google.com/document/d/1lbJFlE8sTkP23Y6ofLxYX0hrh_gWJo2WqzXOyxWgb9c/edit?usp=sharing).

## Change Procedures

These scripts and files are maintained by the Luminoso sales team.

If you are developing something within the se_code package and don't want to have to install/build the package every time you can set an environment variable to the location where your se_code is.

`export PYTHONPATH=/home/you/sales-engineering-code`

This may or many not be necessary if you use python setup.py develop as your
installation command. If your code changes are not running correctly, try
the export command.

Files that you put in the `se_code` directory can be run directly via your
favorite file-running procedure (ipython, for instance, has `%run <filename>`, and can also be imported via `import se_code.<filename>`. You can also make subdirectories of `se_code`; if you do, creating a blank `__init__.py` file in your subdirectory will allow the same kind of importing, i.e. `import se_code.<subdirectory>.<filename>`.

However: please do not add files directly to the master branch of this repository! Instead, there are a few Git commands that will allow you to put
things into a separate, not-yet-approved "branch", and then request that the
development team review your code.  The basic procedure, annotated with numbers in brackets, is:

```
[1] git checkout master
[2] git pull
[3] git checkout -b <descriptive-branch-name>
[4] <code editing goes here>
[5] git commit -a
[6] git push -u origin <descriptive-branch-name>
```

The idea is that, first, you want to make sure you're in the "master"
branch. [1] (`git status` will tell you what branch you're in, or just typing
`git checkout master` will switch you to it if you're not already there.) You
also want to make sure your master branch is up to date [2], or else you'll end
up in what we call "git hell". (Note: if you do ever find yourself in git hell,
just ask someone for help, because everyone ends up there at some point.)

You then want to create a new branch [3], something like `git checkout -b
code-for-InfoCompuCorp` or `git checkout -b topic-copying-code`. `checkout`
tells git to switch you to that branch; `-b` tells it you're making a new
branch.  Once you're in that branch, you can add or modify code to your heart's
content [4].  When you're done you "commit" your changes: the command in [5]
commits everything that's changed, or alternately you can use `git add <files>`
to add just those files you want to add, and then `git commit` to commit the
ones you've added.  Finally, push it to GitHub [6]. If you've previously put it
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

For Luminoso code style guide see: (https://github.com/LuminosoInsight/dev-docs/blob/master/general/style-guide.md)



