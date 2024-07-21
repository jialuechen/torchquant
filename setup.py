from setuptools import setup, find_packages

setup(
    name="quantorch",
    version="1.0.0",
    description="High-Performance PyTorch Library for Derivatives Pricing",
    long_description=open('README.md').read(),
    author="Jialue Chen",
    author_email="jialuechen@outlook.com",
    url="https://github.com/jialuechen/quantorch",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.1",
        "numpy>=1.19.2",
        "scipy>=1.5.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)