import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cycle_prediction",  # Replace with your own username
    version="1.0.4",
    author="Fadi Baskharon",
    author_email="nzfadi@gmail.com",
    description="A package to train and predict the end of a process from\
        history logs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fazaki/cycle_prediction/tree/master",
    packages=setuptools.find_packages(),
    install_requires=[
        'Keras==2.3.1',
        'Keras-Applications==1.0.8',
        'Keras-Preprocessing==1.1.0',
        'numpy>=1.18.2',
        'pandas>=1.0.3',
        'scipy>=1.3.1',
        'tensorboard>=2.0.1',
        'tensorflow==2.0.1',
        'tensorflow-estimator==2.0.1',
        'wtte>=1.1.1',

    ],
    extras_requires={
        "dev": [
            'matplotlib>=3.2.1',
            'ipykernel>=5.3.4',
            'seaborn>=0.10.0',
            'pytest>=6.0.1',
            'pytest-docs>=0.1.0',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
