"""安装脚本"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="toumanfen",
    version="0.1.0",
    author="TouManFen Team",
    description="投满分 - 中文文本分类系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/toumanfen-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "optimization": [
            "pytorch-quantization",
            "torch-pruning",
        ],
    },
    entry_points={
        "console_scripts": [
            "toumanfen-train=scripts.train_baseline:main",
            "toumanfen-predict=scripts.predict:main",
            "toumanfen-server=src.deployment.api.app:run_server",
        ],
    },
)
