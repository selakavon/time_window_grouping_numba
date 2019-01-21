# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='autofe',
    version='0.1.0',
    description='Autofe package',
    long_description=readme,
    author='Ales Novak',
    author_email='ales.novak@sony.com',
    url='https://github.techsoft.eu.sony.com/benovaka/autofe',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
