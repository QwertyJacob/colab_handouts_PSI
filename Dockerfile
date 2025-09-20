FROM manimcommunity/manim:v0.19.0

USER root
RUN pip install notebook matplotlib-venn

# Install git and clean up
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ARG NB_USER=manimuser
USER ${NB_USER}

COPY --chown=manimuser:manimuser . /manim

# Use root for installing packages
USER root

# Install system packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install notebook matplotlib-venn

# Set cache bust
ARG CACHE_BUST=1

# Clone your repo (as root) into /manim
RUN git clone https://github.com/QwertyJacob/colab_handouts_PSI /manim/colab_handouts_PSI

# Set working directory
WORKDIR /manim

# Switch to Binder default user
USER jovyan

# Optional: expose port if running locally
EXPOSE 8888
