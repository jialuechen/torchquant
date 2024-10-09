from setuptools import setup, find_packages

setup(
    name="torchquantlib",
    version="1.3.0",
    description="High-Performance PyTorch Library for Derivatives Pricing",
    author="Jialue Chen",
    author_email="jialuechen@outlook.com",
    url="https://github.com/jialuechen/torchquantlib",
    packages=find_packages(),
    extras_require={
    'docs': ['sphinx', 'sphinx_rtd_theme'],
},
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "geomloss"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)