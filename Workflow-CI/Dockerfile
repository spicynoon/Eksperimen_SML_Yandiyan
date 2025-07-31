FROM continuumio/miniconda3

# Copy environment ke image & install
COPY Workflow-CI/conda.yaml /tmp/conda.yaml
RUN conda env create -f /tmp/conda.yaml \
    && conda clean -afy

# Copy script dan data preprocessing
COPY Workflow-CI/modelling.py /app/modelling.py
COPY Workflow-CI/drybean_preprocessing /app/drybean_preprocessing

WORKDIR /app

# Aktifkan env saat start container
SHELL ["conda", "run", "-n", "drybean-env", "/bin/bash", "-c"]

CMD ["python", "modelling.py"]
