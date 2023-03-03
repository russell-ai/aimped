import setuptools

setuptools.setup(
    name="aimped",
    version="0.0.1",
    author="Russell C.",
    author_email="russell@aimped.com", # optional
    description="A small NLP example package",
    long_description="A small NLP example package with nlp functions",
    long_description_content_type="text/markdown",
    url="https://dev.ml-hub.nioyatechai.com/", # optional
    packages=setuptools.find_packages(),
    install_requires=[ 'nltk', 'numpy', 'pandas' ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
