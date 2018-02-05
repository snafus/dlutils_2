from setuptools import setup

setup(name='dlutils',
      version='0.1.1',
      description='simple set of helpers for DL ',
      url='https://github.com/snafus/dlutils',
      #author='',
      #author_email='flyingcircus@example.com',
      #license='MIT',
      packages=['dlutils'],
      install_requires = [
             'theano',
             'numpy',
             'keras',
             'sklearn',
             'bcolz'
            ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)


