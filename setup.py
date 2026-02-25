#!/usr/bin/env python
"""Setup script for MolFM-Lite"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split("\n")
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]
else:
    requirements = []

setup(
    name="molfm-lite",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-Modal Molecular Foundation Model for Property Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/molfm-lite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "aws": [
            "boto3>=1.28.0",
            "sagemaker>=2.150.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "molfm-pretrain=scripts.pretrain:main",
            "molfm-evaluate=scripts.evaluate:main",
            "molfm-download=scripts.download_data:main",
        ],
    },
)
