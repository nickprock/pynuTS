from setuptools import setup

exec(open('pynuTS/version.py').read())
setup(
   name='pynuTS',
   version=__version__,
   description='A python library for Time Series based on IAML blog article',
   license="LICENSE.txt",
   long_description=open('README.md').read(),
   author='Nicola Procopio',
   author_email='nico.pro412@gmail.com',
   url="https://github.com/nickprock/pynuTS",
   packages=['pynuTS'],  #same as name
   install_requires=['pandas', 'numpy', 'tqdm', 'dtw'], #external packages as dependencies
)