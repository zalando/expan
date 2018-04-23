#!/usr/bin/env python

import re
from setuptools import setup, find_packages

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

try:
    requirements = parse_requirements('requirements.txt')
except OSError:
	requirements = []

with open('README.rst') as readme_file:
	readme = readme_file.read()

with open('CHANGELOG.rst') as history_file:
	history = history_file.read()

with open('expan/core/version.py', 'r') as fd:
	version = re.search(
		r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
		fd.read(),
		re.MULTILINE).group(1)

if not version:
	raise RuntimeError('Cannot find version information')

test_requirements = [
    # the latest pytest 3.1.0 will lead to failure of some unit tests in python 2.7
	'pytest==3.0.7'
]

setup(
	name='expan',
	version=version,
	description="Experiment Analysis Library",
	long_description=readme + '\n\n' + history,
	author="Zalando SE",
	author_email='octopus@zalando.de',
	url='https://github.com/zalando/expan',
	packages=find_packages(),
	package_dir={'expan': 'expan'},
	include_package_data=True,
	install_requires=requirements,
	license="MIT",
	zip_safe=False,
	keywords='expan',
	entry_points={
		'console_scripts': [
			'expan = expan.cli.cli:main'
		]
	},
	classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Developers',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Natural Language :: English',
		"Programming Language :: Python :: 2",
		# 'Programming Language :: Python :: 2.6',
		'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3',
		# 'Programming Language :: Python :: 3.3',
		'Programming Language :: Python :: 3.4',
		'Programming Language :: Python :: 3.5',
	],
	test_suite='tests',
	tests_require=test_requirements
)
