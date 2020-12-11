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
            'lumi_add_concept_relations_to_project = se_code.add_concept_relations_to_project:main',
            'lumi_capitalize_saved_concepts = se_code.capitalize_saved_concepts:main',
            'lumi_copy_shared_concepts = se_code.copy_shared_concepts:main',
            'lumi_create_train_test_split.py = se_code.create_train_test_split:main',
            'lumi_doc_downloader = se_code.doc_downloader:main',
            'lumi_format_multiple_text_fields = se_code.format_multiple_text_fields:main',
            'lumi_get_all_scoredrivers = se_code.get_all_score_drivers:main',
            'lumi_ignore_terms = se_code.ignore_terms:main',
            'lumi_list_metadata = se_code.list_metadaa:main',
            'lumi_load_shared_concepts = se_code.load_shared_concepts:main',
            'lumi_onsite_list_users = se_code.onsite_list_users:main',
            'lumi_onsite_usage = se_code.onsite_usage:main',
            'lumi_project_migration = se_code.project_migration:main',
            'lumi_score_drivers = se_code.score_drivers:main',
            'lumi_bi_tool_export = se_code.bi_tool_export:main',

        ]},
)
