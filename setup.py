from setuptools import setup, find_packages

setup(
    name="se_code",
    version='0.14.0',
    maintainer='Luminoso Technologies, Inc.',
    maintainer_email='dev@luminoso.com',
    license="LICENSE",
    url='http://github.com/LuminosoInsight/sales-engineering-code',
    platforms=["any"],
    description=("Code for sales engineering, particularly for code that will be given to customers"),
    packages=find_packages(),
    install_requires=[
        'luminoso_api', 'click', 'scipy', 'pack64', 'numpy', 'scikit-learn',
        'redis', 'flask', 'networkx', 'praw', 'pandas', 'psycopg2-binary'
        ],
    entry_points={
        'console_scripts': [
            'lumi-add-concept-relations-to-project = se_code.add_concept_relations_to_project:main',
            'lumi-add-outlier-concepts-to-list = se_code.add_outlier_concepts_to_list:main',
            'lumi-add-sentiment_as_driver = se_code.add_sentiment_as_driver:main',
            'lumi-bi-tool-export = se_code.bi_tool_export:main',
            'lumi-boilerplate_remover = se_code.boilerplate_remover:main',
            'lumi-capitalize-saved-concepts = se_code.capitalize_saved_concepts:main',
            'lumi-copy-project-with-filter = se_code.copy_project_with_filter:main',
            'lumi-copy-shared-concepts = se_code.copy_shared_concepts:main',
            'lumi-copy-shared-views = se_code.copy_shared_views:main',
            'lumi-create-daylight-project-from-csv = se_code.create_daylight_project_from_csv:main',
            'lumi-create-train-test-split = se_code.create_train_test_split:main',
            'lumi-doc-downloader = se_code.doc_downloader:main',
            'lumi-format-multiple-text-fields = se_code.format_multiple_text_fields:main',
            'lumi-format-survey = se_code.format_survey:main',
            'lumi-list-metadata = se_code.list_metadata:main',
            'lumi-manage-concept-lists = se_code.manage_concept_lists:main',
            'lumi-manage-concepts = se_code.manage_concepts:main',
            'lumi-project-migration = se_code.project_migration:main',
            'lumi-sentiment = se_code.sentiment:main',
            'lumi-score-drivers = se_code.score_drivers:main',
            'lumi-sentiment-colorization = se_code.sentiment_colorization:main',
            'lumi-wheel-info = se_code.wheel_info:main',
        ]},
)
