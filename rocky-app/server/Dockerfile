# Use an official Python runtime based on Debian 10 ("buster") as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Add the source code to app
ADD .. /src

# Run commands to update Conda and set up the environment
RUN conda update -n base -c defaults conda

# Create a new Conda environment "myenv" and install packages
RUN conda install -c conda-forge python=3.9.16 pandas numpy openpyxl scipy plotly scikit-learn statsmodels flask flask-cors flask-sqlalchemy "pymc>=4" arviz

# Install packages from req4.txt
# RUN conda install -n myenv -c conda-forge 

# Make RUN commands use the new environment
SHELL ["/opt/conda/envs/myenv/bin/python", "-m", "server/app.py"]
