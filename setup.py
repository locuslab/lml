from setuptools import find_packages, setup

setup(
    name='lml',
    version='0.0.1',
    description="The limited multi-label projection layer.",
    author='Brandon Amos',
    author_email='brandon.amos.cs@gmail.com',
    platforms=['any'],
    license="MIT",
    url='https://github.com/locuslab/lml',
    py_modules=['lml'],
    install_requires=[
        'numpy>=1<2',
        'semantic_version',
    ]
)
