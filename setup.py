from setuptools import setup, find_packages

setup(
    name="torchquantlib",
    version="1.2.0",
    description="High-Performance Torch Library for Derivatives Pricing",
    author="Jialue Chen",
    author_email="jialuechen@outlook.com",
    url="https://github.com/jialuechen/torchquantlib",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scipy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)