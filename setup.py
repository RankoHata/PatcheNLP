# -*- coding: utf-8 -*-
"""
> python setup.py sdist build
> twine upload dist/*
"""
from setuptools import setup, find_packages

setup(
    name='PatcheNLP',
    version='0.0.3',
    description='Some tools for NLP.',
    author='RankoHata',
    author_email='san332627946@gmail.com',
    maintainer='RankoHata',
    maintainer_email='san332627946@gmail.com',
    license=' GNU General Public License v3',
    packages = find_packages(),
    include_package_data=True,
    platforms=['all'],
    url='https://github.com/RankoHata/PatcheNLP',
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation',
        'Topic :: Software Development :: Libraries'
    ],
)