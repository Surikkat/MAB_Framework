from setuptools import setup, find_packages

def readme():
  with open('README.md', 'r') as f:
    return f.read()

setup(
  name='BanditLab',
  version='1.0.0',
  author='Surikkat',
  author_email='surikkatik@gmail.com',
  description='Framework for MAB',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='',
  packages=find_packages(),
  install_requires=[''],
  classifiers=[
    'Programming Language :: Python :: 3.12',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  keywords='',
  project_urls={
    'Documentation': 'link'
  },
  python_requires='>=3.12'
)