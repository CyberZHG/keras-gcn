from setuptools import setup

setup(
    name='keras-gcn',
    version='0.7',
    packages=['keras_gcn'],
    url='https://github.com/CyberZHG/keras-gcn',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='Graph convolutional layers',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'Keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
