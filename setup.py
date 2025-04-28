from setuptools import setup, find_packages

setup(
    name = "fluxRecovery",                 
    version = "0.1",                 
    packages = find_packages(),      
    install_requires = ["pandas", "xarray", "scikit-learn"],
    author = "Karan Bhalla",            
    description = "A package for eddy flux data post processing",  
    long_description = open("README.md").read(),  
    long_description_content_type = "text/markdown", 
    url = "https://github.com/Aran203/fluxRecovery",  
    classifiers = [                  
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',       
    license = "MIT",                               
    include_package_data = True,   
)