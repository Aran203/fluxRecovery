from setuptools import setup, find_packages

setup(
    name = "fluxpy",                 
    version = "0.1",                 
    packages = find_packages(),      
    install_requires = ["pandas"],
    author = "Karan Bhalla",            
    description = "A package for eddy flux data post processing",  
    long_description = open("README.md").read(),  
    long_description_content_type = "text/markdown", 
    url = "https://github.com/Aran203/fluxpy",  
    classifiers = [                  
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',       
    license = "TBD",                    # Add licensing              
    include_package_data = True,   
)