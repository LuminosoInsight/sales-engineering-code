from setuptools import setup, find_packages

setup(
    name="se_code",
    version='0.2',
    maintainer='Luminoso Technologies, Inc.',
    maintainer_email='dev@luminoso.com',
    license="LICENSE",
    url='http://github.com/LuminosoInsight/sales-engineering-code',
    platforms=["any"],
    description=("Code for sales engineering, particularly for code that will be given to customers"),
    packages=find_packages(),
    install_requires=[
        'luminoso_api', 'click', 'scipy', 'pack64', 'numpy', 'scikit-learn', 
        'redis', 'flask', 'networkx', 'praw', 'pandas'
        ],
    entry_points={
        'console_scripts': [
            'lumi-add-concept-relations-to-project = se_code.add_concept_relations_to_project:main',
            'lumi-capitalize-saved-concepts = se_code.capitalize_saved_concepts:main',
            'lumi-copy-shared-concepts = se_code.copy_shared_concepts:main',
            'lumi-create-train-test-split = se_code.create_train_test_split:main',
            'lumi-doc-downloader = se_code.doc_downloader:main',
            'lumi-format-multiple-text-fields = se_code.format_multiple_text_fields:main',
            'lumi-ignore-terms = se_code.ignore_terms:main',
            'lumi-list-metadata = se_code.list_metadata:main',
            'lumi-load-shared-concepts = se_code.load_shared_concepts:main',
            'lumi-onsite-list-users = se_code.onsite_list_users:main',
            'lumi-onsite-usage = se_code.onsite_usage:main',
            'lumi-project-migration = se_code.project_migration:main',
            'lumi-score-drivers = se_code.score_drivers:main',
            'lumi-bi-tool-export = se_code.bi_tool_export:main',

        ]},
)
