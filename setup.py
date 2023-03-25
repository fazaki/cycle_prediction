import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cycle_prediction",  # Replace with your own username
    version="1.2.1",
    author="Fadi Baskharon",
    author_email="nzfadi@gmail.com",
    description="A package to train and predict the end of a process from\
        history logs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fazaki/cycle_prediction/tree/master",
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow==2.11.1',
        'Keras==2.3.1',
        'Keras-Applications==1.0.8',
        'Keras-Preprocessing==1.1.0',
        'matplotlib>=3.2.1',
        'numpy>=1.18.2',
        'pandas>=1.0.0',
        'scikit-learn>=0.22.2.post1',
        'scipy>=1.3.1',
        'seaborn>=0.10.0',
        'wtte>=1.1.1',
    ],
    extras_require={
        "dev": [
            'pylint==2.6.0',
            'twine>=3.2.0',
            'pytest==6.0.1',
            'pytest-docs==0.1.0'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
