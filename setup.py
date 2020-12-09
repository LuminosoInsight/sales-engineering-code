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
            'lumi-add-concept-relations-to_project = se_code.add_concept_relations_to_project:main',
            'lumi-copy-shared-concepts = se_code.copy_shared_concepts:main',
            'lumi-doc-downloader = se_code.doc_downloader:main'
        ]},
)
