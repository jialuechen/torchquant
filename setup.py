import setuptools

setuptools.setup(
    name='quantorch',
    version='0.1.0',
    packages=setuptools.find_packages(),
    install_requires=[
        "pytorch==2.0",
        "pandas==2.0.3",
        "networkx==3.1"
    ],
    url='https://github.com/jialuechen/quantorch',
    license='MIT',
    author='Jialue Chen',
    author_email='jialuechen@outlook.com',
    description='A PyTorch-Based Python Library for Quantitative Finance'
)