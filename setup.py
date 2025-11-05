from setuptools import setup, find_packages

setup(
    name="lwe-cryptography-thesis",
    version="1.0.0",
    author="Your Name",
    description="LWE Cryptography Optimization with Image Compression",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "pillow>=10.0.0",
        "matplotlib>=3.7.2",
        "pandas>=2.0.3",
        "scipy>=1.11.1",
    ],
)
