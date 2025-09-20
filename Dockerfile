FROM manimcommunity/manim:v0.19.0

USER root

RUN pip install notebook matplotlib-venn

# Install git and clean up

RUN apt-get update && \ 
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git ca-certificates && \
    rm -rf /var/lib/apt/lists/* 

ARG CACHE_BUST=1 

RUN git clone https://github.com/QwertyJacob/colab_handouts_PSI

RUN chown -R manimuser:manimuser /manim/colab_handouts_PSI

ARG NB_USER=manimuser 
    
USER ${NB_USER} 

COPY --chown=manimuser:manimuser . /manim

