from setuptools import find_packages, setup

setup(
    name="customer_segmentation",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=1.3.5",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.1",
        "seaborn>=0.11.2",
        "plotly>=5.6.0",
        "streamlit>=1.10.0",
        "streamlit-plotly-events>=0.0.6",
        "streamlit-option-menu>=0.3.2",
        "joblib>=1.1.0",
        "python-dotenv>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.3.0",
            "flake8>=4.0.1",
        ],
    },
    author="Abdiwahid Hussein Ali",
    author_email="maqbuul@outlook.com",
    description="A customer segmentation project using RFM analysis",
    keywords="customer segmentation, RFM, clustering, marketing",
    url="https://github.com/codeandcharts/customer_segmentation_project",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Business/Marketing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Business :: CRM",
    ],
    python_requires=">=3.8",
)
