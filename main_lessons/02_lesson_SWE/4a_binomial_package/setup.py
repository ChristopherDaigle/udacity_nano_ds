from setuptools import setup

setup(name='daigle_dist',
      version='0.1',
      description='Gaussian and Binomial Distributions',
      packages=['daigle_dist'],
      author='Christopher Daigle',
      author_email='pcjdaigle@gmail.com',
      zip_safe=False)


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='daigle_dist',
      version="0.1.1",
      author="Christopher Daigle",
      author_email="pcjdaigle@gmail.com",
      description="Gaussian and Binomial Distributions",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/ChristopherDaigle/udacity_nano_ds/tree/master/02_lesson_SWE/4a_binomial_package",
      packages=['daigle_dist'],
      classifiers=["Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent"],
      python_requires='>=3.6',
      zip_safe=False)
