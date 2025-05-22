# ...existing code...
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torchquantlib",
    version="1.3.1",
    description="High-Performance PyTorch Library for Derivatives Pricing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jialue Chen",
    author_email="jialuechen@outlook.com",
    url="https://github.com/jialuechen/torchquantlib",
    packages=find_packages(),
    license="Apache-2.0",
    keywords="quantitative finance, derivatives, pytorch, option pricing, risk management",
    include_package_data=True,
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
# ...existing code...