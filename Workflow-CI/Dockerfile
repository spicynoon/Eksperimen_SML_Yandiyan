FROM continuumio/miniconda3

# Copy environment spesifik ke image
COPY Workflow-CI/conda.yaml /tmp/conda.yaml
RUN conda env create -f /tmp/conda.yaml

# Copy script dan data preprocessing ke dalam image
COPY Workflow-CI/modelling.py /app/modelling.py
COPY Workflow-CI/drybean_preprocessing /app/drybean_preprocessing

WORKDIR /app
ENV PATH /opt/conda/envs/drybean-env/bin:$PATH
CMD ["python", "modelling.py"]
