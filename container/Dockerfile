# syntax=docker/dockerfile:1
FROM continuumio/miniconda3:23.10.0-1  AS builder

RUN apt-get update && apt-get install -y \
build-essential
WORKDIR /tmp/
RUN wget https://mirrors.sarata.com/gnu/gsl/gsl-2.7.tar.gz && \
    tar -zxvf gsl-2.7.tar.gz && \
    cd gsl-2.7 && \
    ./configure && \
    make -j 8 && \
    make install && \
    cd .. && \
    rm -rf gsl-2.7.tar.gz gsl-2.7

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -a && conda init bash && echo "conda activate $(head -1 environment.yml | cut -d' ' -f2)" >> ~/.bashrc && \
    rm environment.yml

FROM continuumio/miniconda3:23.10.0-1
RUN apt-get update && apt-get install -y \
build-essential \
ghostscript \
git \
git-annex \
openssh-client

RUN mkdir /app

COPY --from=builder /opt/conda/envs/ClusteredNetwork_pub /opt/conda/envs/ClusteredNetwork_pub
COPY --from=builder /usr/local/lib /usr/local/lib


RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" >> /root/.bashrc && \
    echo "export CFLAGS="-I/usr/local/include"" >> /root/.bashrc && \
    echo "export LDFLAGS="-L/usr/local/lib"" >> /root/.bashrc && \
    echo "conda activate ClusteredNetwork_pub" >> ~/.bashrc

WORKDIR /app/