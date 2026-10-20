# ssm (lindermanlab/ssm) has no PyPI release and builds a Cython extension, so it runs here, not in the workspace venv.
# Build:  docker build -t iohmm-ssm-oracle ml/tests/hmm/iohmm/oracle/ssm
# Run:    docker run --rm -v "<abs path to ml/tests/hmm/iohmm/fixtures>:/out" iohmm-ssm-oracle
FROM python:3.10.14-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy==1.26.4" "scipy==1.11.4" "autograd==1.6.2" "numba==0.58.1" "cython==0.29.37" \
    "scikit-learn==1.3.2" "tqdm==4.66.4" "matplotlib==3.8.4"
# eb6c8aa (2025-05-09) is the current master head; ssm has no tags
RUN git clone https://github.com/lindermanlab/ssm.git /opt/ssm && cd /opt/ssm && git checkout eb6c8aa \
    && pip install --no-cache-dir --no-build-isolation -e /opt/ssm

COPY make_reference.py /opt/make_reference.py
CMD ["python", "/opt/make_reference.py", "/out/ssm_reference.npz"]
