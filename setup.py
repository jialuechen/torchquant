import setuptools

setuptools.setup(
    name='quantorch',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=[
        "pytorch==1.10.2",
    ],
    url='https://github.com/jialuechen/quantorch',
    license='MIT',
    author='Jialue Chen',
    author_email='jialuechen@outlook.com',
    description='A PyTorch-Based Python Library for Quantitative Finance'
)