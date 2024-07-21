from setuptools import setup, find_packages

setup(
    name='quantorch',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'numpy>=1.19.2',
        'scipy>=1.6.0'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A quantitative finance library leveraging PyTorch.',
    url='https://github.com/yourusername/quantorch',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)