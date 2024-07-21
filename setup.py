from setuptools import setup, find_packages

setup(
    name="quantorch",
    version="0.1.0",
    description="Quantitative Finance Library using PyTorch",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/jialuechen/quantorch",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.1",
        "numpy>=1.19.2",
        "scipy>=1.5.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)