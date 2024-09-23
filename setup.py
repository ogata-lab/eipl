from setuptools import setup, find_packages

DESCRIPTION = "EIPL: Embodied Intelligence with Deep Predictive Learning"
NAME = "eipl"
AUTHOR = "Hiroshi Ito"
AUTHOR_EMAIL = "it.hiroshi.o@gmail.com"
URL = "https://github.com/ogata-lab/eipl"
LICENSE = "MIT License"
DOWNLOAD_URL = "https://github.com/ogata-lab/eipl"
VERSION = "1.1.1"
PYTHON_REQUIRES = ">=3.8"

"""
INSTALL_REQUIRES = [
    'pytorch>=1.9.0',
    'matplotlib>=3.3.4',
    'numpy >=1.20.3',
    'matplotlib>=3.3.4',
    'scipy>=1.6.3',
    'scikit-learn>=0.24.2',
]
"""

"""
with open('README.rst', 'r') as fp:
    readme = fp.read()
with open('CONTACT.txt', 'r') as fp:
    contacts = fp.read()
long_description = readme + '\n\n' + contacts
"""


setup(
    name=NAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    # long_description=long_description,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    # python_requires=PYTHON_REQUIRES,
    # install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
)
