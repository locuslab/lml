from setuptools import find_packages, setup

setup(
    name='lml',
    version='0.0.1',
    description="TODO",
    author='Brandon Amos',
    author_email='brandon.amos.cs@gmail.com',
    platforms=['any'],
    license="Apache 2.0",
    url='https://github.com/locuslab/lml',
    packages=find_packages(),
    install_requires=[
        'numpy>=1<2',
    ]
)
