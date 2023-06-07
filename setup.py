#Author: Russell C.
#Date: 2023-03-20
#Copyright: (c) 2023 aimped
#Description: This is the setup file for the aimped library.

import setuptools
from aimped.version import __version__

setuptools.setup(
    name="aimped",
    version= __version__,
    packages=setuptools.find_packages(),
    install_requires=[ 
                        'nltk',
                        'numpy',
                        'pandas',
                        'scikit-learn',
                        "seqeval",
                        "pydantic",
                        ],
    author="Russell C.",
    author_email="russell@aimped.com",
    maintainer="aimped",
    maintainer_email="contact@aimped.com",
    description="Aimped is a unique library that provides classes and functions for only exclusively business-tailored AI-based NLP models.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://dev.aimped.ai/", 
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

