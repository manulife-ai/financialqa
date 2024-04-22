from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Financial QABot pdf parser and ingestion pipeline'
LONG_DESCRIPTION = '''Parses text and tables from financial reports and ingests
                    them into an Azure Search index for Financial 
                    Question-Answering using Retrieval Augmented Generation'''

# Setting up
setup(
        # the name must match the folder name 'verysimplemodule'
        name="financialqa", 
        version=VERSION,
        author="Messiah Ataey",
        author_email="Messiah_Ataey@manulife.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        # packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'

        packages=find_packages('financialqa'),
        package_dir={'': 'src'},

        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
