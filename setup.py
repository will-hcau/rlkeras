from setuptools import setup
from setuptools import find_packages


setup(name='rlkeras',
      version='0.1',
      description='Deep Reinforcement Learning using Keras',
      author='Will AU',
      author_email='will.hcau@gmail.com',
      url='https://github.com/will-hcau/rlkeras.git',
      license='MIT',
      install_requires=['keras>=2.0.7'],
      extras_require={
          'gym': ['gym'],
      },
      packages=find_packages())
