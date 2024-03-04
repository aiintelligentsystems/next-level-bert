# syntax=docker/dockerfile:1.5.2

# This Dockerfile produces a container with all dependencies installed into an environment called "research"
# Additionally, the container has a full-fledged micromamba installation, which is a faster drop-in replacement for conda
# When inside the container, you can install additional dependencies with `mamba install <package>`, e.g. `mamba install scipy`
# The actual installation is done by micromamba, we have simply provided an alias to link the mamba command to micromamba

# The syntax line above is crucial to enable variable expansion for type=cache=mount commands

# We can use the OS_PREFIX build arg to choose between ubi8 and ubuntu as base image (for amd64 processor architecture)
ARG OS_SELECTOR=ubi8

# Load micromamba container to copy from later
FROM --platform=$TARGETPLATFORM mambaorg/micromamba:1.5.1 as micromamba

####################################################
################ BASE IMAGES #######################
####################################################

# -----------------
# base image for amd64
# -----------------
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-runtime-ubi8 as amd64ubi8
# Install compiler for .compile() with PyTorch 2.0 and nano for devcontainers
RUN yum install -y git gcc gcc-c++ nano && yum clean all
# Copy lockfile to container
COPY conda-lock.yml /locks/conda-lock.yml

# -----------------
# devcontainer base image for amd64 using Ubuntu
# SLURM + pyxis has a bug on our cluster, where the automatic activation of the conda environment fails if the base image is ubuntu
# But Ubuntu works better for devcontainers than ubi8
# So we use Ubuntu for devcontainers and ubi8 for actual deployment on the cluster
# -----------------
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as amd64ubuntu
# Install compiler for .compile() with PyTorch 2.0 and nano for devcontainers
RUN apt-get update && apt-get install -y git gcc g++ nano openssh-client && apt-get clean
# Copy lockfile to container
COPY conda-lock.yml /locks/conda-lock.yml

# -----------------
# base image for ppc64le
# -----------------
FROM --platform=linux/ppc64le nvidia/cuda:11.8.0-cudnn8-runtime-ubi8 as ppc64leubi8
# Install compiler for .compile() with PyTorch 2.0
RUN yum install -y gcc gcc-c++ && yum clean all
# Copy ppc64le specififc lockfile to container
COPY ppc64le.conda-lock.yml /locks/conda-lock.yml


###########################################################
################ Intermediate IMAGE #######################
###########################################################
# -----------------
# build image - we choose the correct base image based on the target architecture and OS
# -----------------
ARG TARGETARCH
FROM ${TARGETARCH}${OS_SELECTOR} as nvidia-cuda-with-micromamba

# ---------
# From https://github.com/mamba-org/micromamba-docker#adding-micromamba-to-an-existing-docker-image
# The commands below add micromamba to an existing image to give the capability to ad-hoc install new dependencies
USER root

# if your image defaults to a non-root user, then you may want to make the
# next 3 ARG commands match the values in your image. You can get the values
# by running: docker run --rm -it my/image id -a
ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
ENV XDG_CACHE_HOME="/home/mamba/.cache/"

# to prevent build for ppc64le from getting stuck https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html?highlight=create#hangs-during-install-in-qemu
ENV G_SLICE="always-malloc"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
# Optional: if you want to customize the ENTRYPOINT and have a conda
# environment activated, then do this:
# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "my_entrypoint_program"]

# You can modify the CMD statement as needed....
CMD ["/bin/bash"]
# ------------


############################################################
######### Install dependcies in seperate image #############
############################################################
FROM nvidia-cuda-with-micromamba as installed-dependencies

# Necessary to prevent permission error when micromamba tries to install pip dependencies from lockfile
USER root
RUN chown $MAMBA_USER:$MAMBA_USER /locks/
RUN chown $MAMBA_USER:$MAMBA_USER /locks/conda-lock.yml
USER $MAMBA_USER

ARG TARGETPLATFORM
# Use line below to debug if cache is correctly mounted
# RUN --mount=type=cache,target=$MAMBA_ROOT_PREFIX/pkgs,id=conda-$TARGETPLATFORM,uid=$MAMBA_USER_ID,gid=$MAMBA_USER_GID ls -al /opt/conda/pkgs
# Install dependencies from lockfile into environment, cache packages in /opt/conda/pkgs
RUN --mount=type=cache,target=$MAMBA_ROOT_PREFIX/pkgs,id=conda-$TARGETPLATFORM,uid=$MAMBA_USER_ID,gid=$MAMBA_USER_GID \
    micromamba install --name base --yes --file /locks/conda-lock.yml 

# Install optional tricky pip dependencies that do not work with conda-lock
# --no-deps --no-cache-dir to prevent conflicts with micromamba, might have to remove it depending on your use case
# RUN micromamba run -n research pip install example-dependency --no-deps --no-cache-dir


###############################################
############# FINAL IMAGE #####################
###### For smaller resulting image size #######
###############################################
FROM nvidia-cuda-with-micromamba as final

# Get ONLY the micromamba environment from the previous image, chmod 777 s.t. any user can use micromamba
COPY --from=installed-dependencies --chmod=777 /opt/conda /opt/conda
COPY --from=installed-dependencies --chmod=777 /locks/conda-lock.yml  /locks/conda-lock.yml

# Grant some useful permissions
USER root
# Give user permission to gcc
RUN chown $MAMBA_USER:$MAMBA_USER /usr/bin/gcc
# Provide mamba alias for micromamba
RUN echo "alias mamba=micromamba" >> /usr/local/bin/_activate_current_env.sh
# Give permission to everyone for e.g. caching
RUN mkdir /home/${MAMBA_USER}/.cache && chmod -R 777 /home/${MAMBA_USER}/.cache/
USER $MAMBA_USER

# Set conda-forge as default channel (otherwise no default channel is set)
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba config prepend channels conda-forge --env
# Disable micromamba banner at every command
RUN micromamba config set show_banner false --env