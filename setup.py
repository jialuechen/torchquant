import setuptools

setuptools.setup(
    name='quantorch',
    version='0.2.0',
    packages=setuptools.find_packages(),
    install_requires=[
        "pytorch==2.3.1",
        "pandas==2.2.2",
        "networkx==3.3",
        "py_vollib==1.0.3"
    ],
    url='https://github.com/jialuechen/quantorch',
    license='MIT',
    author='Jialue Chen',
    author_email='jialuechen@outlook.com',
    description='A PyTorch-Based Python Library for Quantitative Finance'
)