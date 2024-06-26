from setuptools import setup, find_packages

setup(
    name="decker",
    version="2.0.0",
    author="Will Decker",
    author_email="deckerwill7@gmail.com",
    description="Modularized code for Will Decker's Honors Thesis",
    url="https://github.com/w-decker/decker_honors_module",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
    install_requires=['nilearn', 
                      'matplotlib',
                      'numpy']

)