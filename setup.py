from setuptools import setup, find_packages

setup(
    name="se_code",
    version = '0.1',
    maintainer='Luminoso Technologies, Inc.',
    maintainer_email='dev@luminoso.com',
    license = "LICENSE",
    url = 'http://github.com/LuminosoInsight/sales-engineering-code',
    platforms = ["any"],
    description = ("Code for sales engineering, particularly for code that "
                   "will be given to customers"),
    packages=find_packages(),
    install_requires=[
        'luminoso_api'
        ],
    entry_points={
        'console_scripts': [
            'lumi-doc-download = se_code.doc_downloader:main',
            'lumi-topic-copier = se_code.topic_copier:main',
            'lumi-topic-generator = se_code.topic_generator:main',
    ]},
)
