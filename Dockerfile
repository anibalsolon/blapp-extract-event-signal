FROM continuumio/miniconda3 AS base

RUN conda install -y -c conda-forge numpy scipy pandas && conda clean -y --all
RUN pip install nibabel
RUN mkdir /scratch
WORKDIR /scratch