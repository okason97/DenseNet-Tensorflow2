import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='densenet',  

     version='0.1',

     author="Gaston Gustavo Rios",

     author_email="okason1997@hotmail.com",

     description="A densenet implementation using tensorflow2",

     long_description=long_description,

     long_description_content_type="text/markdown",

     url="https://github.com/okason97/DenseNet-Tensorflow2",

     packages=setuptools.find_packages(),
     
     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ],

)
