from setuptools import setup, find_packages

setup(
    name="Sales Engineering Code",
    version = '0.1',
    maintainer='Luminoso, LLC',
    maintainer_email='dev@lumino.so',
    license = "LICENSE",
    url = 'http://github.com/LuminosoInsight/sales-engineering-code',
    platforms = ["any"],
    description = ("Code for sales engineering, particularly for code that "
                   "will be given to customers"),
    packages=find_packages(),
    install_requires=[
        ],
    entry_points={
    },
)
