"""
Setup script for BulkRNA Agent
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bulkrna-agent",
    version="0.1.0",
    author="Shivaprasad Patil",
    description="AI-powered bulk RNA-seq analysis agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shivaprasad-patil/BulkRNA-Agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "gradio>=4.0.0",
        "requests>=2.31.0",
        "pydeseq2>=0.4.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "bulkrna-agent=bulkrna_agent.web_interface:main",
        ],
    },
)
