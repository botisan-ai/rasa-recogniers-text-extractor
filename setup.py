#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'rasa>=3'
]

test_requirements = [ ]

setup(
    author="Simon Liang",
    author_email='simon@x-tech.io',
    python_requires='>=3.7,<3.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="A Rasa NLU module for extracting entities using Microsoft's Recognizers Text library",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='rasa-recognizers-text-extractor',
    name='rasa-recognizers-text-extractor',
    packages=find_packages(include=['rasa_recognizers_text_extractor', 'rasa_recognizers_text_extractor.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/botisan-ai/rasa-recognizers-text-extractor',
    version='0.1.0',
    zip_safe=False,
)
