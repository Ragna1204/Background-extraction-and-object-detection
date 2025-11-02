"""
Setup script for Background Extraction and Object Detection package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="background-motion-detection",
    version="1.0.0",
    author="Ragna1204",
    author_email="",
    description="A Python-based computer vision application for real-time background subtraction and motion detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ragna1204/Background-extraction-and-object-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "gui": [
            "tkinter",
        ],
    },
    entry_points={
        "console_scripts": [
            "motion-detect=cli:main",
            "bg-detect=cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="computer-vision opencv motion-detection background-subtraction",
    project_urls={
        "Bug Reports": "https://github.com/Ragna1204/Background-extraction-and-object-detection/issues",
        "Source": "https://github.com/Ragna1204/Background-extraction-and-object-detection",
    },
)
