import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='densenet',  

     version='0.1',

     author="Gaston Gustavo Rios, Ulises Jeremias Cornejo Fandos",

     author_email="okason1997@hotmail.com, ulisescf.24@gmail.com",

     description="A densenet implementation using tensorflow2",

     long_description=long_description,

     long_description_content_type="text/markdown",

     url="https://github.com/okason97/DenseNet-Tensorflow2",

     packages=setuptools.find_packages(),
     
     install_requires=[
        'tensorflow==2.0.0-alpha0',
        'numpy',
     ],

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ],

)
