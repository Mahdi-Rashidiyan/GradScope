from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gradscope",
    version="0.1.0",
    author="Mahdi",
    description="Training dynamics diagnostics for PyTorch — gradient norms, cosine similarity, and CKA in one line.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mahdi-Rashidiyan/GradScope",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "matplotlib>=3.4",
        "numpy>=1.20",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
