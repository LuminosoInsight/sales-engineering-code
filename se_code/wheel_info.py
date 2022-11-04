
help_string = """Luminoso Wheel File Information

The wheel file from Luminoso comes with the following commands

lumi-add-concept-relations-to-project 
    Use shared concepts to tag documents and create a new project.
lumi-bi-tool-export
    Business Intelligence Export. Creates CSV files for use with business 
    intelligence tools like Tableau, Microstrategy and PowerBI to name a few.
lumi-capitalize-saved-concepts
    Capitalize all the concepts in all the shared concept lists.
lumi-copy-project-with-filter
    Copy a project to a new project using a filter to exclude 
    specific documents based on metadata fields.
lumi-copy-shared-concepts
    Copy shared concepts from one project to another.
lumi-copy-shared-views
    Copy shared views and the shared concept lists associated with those views to another project.
lumi-create-daylight-project-from-csv
    Create a daylight project from a csv file. The CSV has the exact same
    format required by the UI and allows programmatic project creation.
lumi-create-train-test-split
    Will split a Compass voting classifier file into a training and test set.
lumi-doc-downloader
    Download documents from a project. This script can also tag each document
    with the shared concept lists with which it is associated.
lumi-format-multiple-text-fields
    This will take a CSV file with multiple text fields and combine them into
    a single text field with a new metadata column
lumi-manage-concepts
    Uses the /concepts/manage endpoint to fine-tune the concepts in a
    number of ways: marking texts for exclusion from the build process, 
    marking inherently ignored texts for inclusion, marking two different 
    texts to be treated as the same text,
lumi-list-metadata
    This simply lists all the metadata fields and how many unique values 
    there are in each field.
lumi-load-shared-concepts
    This will upload a file of shared concepts to a daylight project.
lumi-project-migration
    Will copy all projects under a specific workspace to a separate 
    server/workspace
lumi-score-drivers
    This can download score drivers for a project as well as generate 
    output for score drivers over time.

More details on each command can be found by using a --help alone on
the command line.  ie.
    lumi-bi-tool-export --help

All Commands
lumi-add-concept-relations-to-project 
lumi-bi-tool-export
lumi-capitalize-saved-concepts
lumi-copy-project-with-filter
lumi-copy-shared-concepts
lumi-copy-shared-views
lumi-create-daylight-project-from-csv
lumi-create-train-test-split
lumi-doc-downloader
lumi-format-multiple-text-fields
lumi-manage-concepts
lumi-list-metadata
lumi-load-shared-concepts
lumi-project-migration
lumi-score-drivers
"""


def main():
    print(help_string)


if __name__ == '__main__':
    main()
