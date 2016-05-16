#!/usr/bin/env python

from pip.req import parse_requirements
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

try:
	install_reqs = parse_requirements('requirements.txt', session=False)
	requirements = [str(ir.req) for ir in install_reqs]
except:
	requirements = []

with open('README.rst') as readme_file:
	readme = readme_file.read()

with open('HISTORY.rst') as history_file:
	history = history_file.read()

test_requirements = [
	'pytest'
]


class build_ext(_build_ext):
	def finalize_options(self):
		_build_ext.finalize_options(self)
		# Prevent numpy from thinking it is still in its setup process:
		__builtins__.__NUMPY_SETUP__ = False
		import numpy
		self.include_dirs.append(numpy.get_include())


setup(
	name='expan',
	version='0.2.4',
	description="Experiment Analysis Library",
	long_description=readme + '\n\n' + history,
	author="Zalando SE",
	author_email='octopus@zalando.de',
	url='https://github.com/zalando/expan',
	packages=find_packages(),
	package_dir={'expan':
					 'expan'},
	include_package_data=True,
	install_requires=requirements,
	cmdclass={'build_ext': build_ext},
	setup_requires=['numpy'],
	license="MIT",
	zip_safe=False,
	keywords='expan',
	classifiers=[
		'Development Status :: 2 - Pre-Alpha',
		'Intended Audience :: Developers',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Natural Language :: English',
		# "Programming Language :: Python :: 2",
		# 'Programming Language :: Python :: 2.6',
		'Programming Language :: Python :: 2.7',
		# 'Programming Language :: Python :: 3',
		# 'Programming Language :: Python :: 3.3',
		# 'Programming Language :: Python :: 3.4',
		# 'Programming Language :: Python :: 3.5',
	],
	test_suite='tests',
	tests_require=test_requirements
)
