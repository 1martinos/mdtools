from setuptools import setup, find_packages
setup(
        name='mdtools',
        version='0.0.1',
        #url='https://github.com/mypackage.git',
        #author='Author Name',
        #author_email='author@gmail.com',
        description='Personal Utilities',
        packages=find_packages(),    
        install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
    )
