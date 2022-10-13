from setuptools import setup, find_packages

setup(
  name = 'HyperTensors',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Hyperbolic objects and operations - Pytorch',
  author = 'Thibault LSDC',
  author_email = 'thibault.de.chezelles@gmail.com',
  url = 'https://github.com/ThibaultLSDC/HyperTensors',
  install_requires=[
    'torch'
  ]
)