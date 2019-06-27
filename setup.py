from distutils.core import setup
import os

long_desc = """
A python library for building autoencoders with tabular data.
Currently in development.
"""

reqs= [
    'torch',
    'numpy',
    'pandas',
    'tqdm',
    'scikit-learn'
]
version = '0.0.34'

setup(
    name='dfencoder',
    version=f'{version}',
    description='Autoencoder Library for Tabular Data',
    long_description=long_desc,
    author='Michael Klear',
    author_email='michael.r.klear@gmail.com',
    url='https://github.com/alliedtoasters/dfencoder',
    download_url=f'https://github.com/alliedtoasters/dfencoder/archive/v{version}.tar.gz',
    install_requires=reqs,
    packages=['dfencoder']
)
