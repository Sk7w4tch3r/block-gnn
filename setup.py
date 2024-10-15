from setuptools import setup, find_packages

setup(
    name="block-gnn",
    version="0.1",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[],
    author="Hesam Damghanian",
    author_email="hesam.damghanian@ucalgary.ca",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sk7w4tch3r/block-gnn",
    python_requires='>=3.11',
)
