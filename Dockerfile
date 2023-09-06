FROM ghcr.io/osgeo/gdal:ubuntu-full-3.7.1

# FROM mcr.microsoft.com/planetary-computer/python

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libpq-dev \
    ca-certificates \
    build-essential \
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

ADD . /code

ADD https://deppcpublicstorage.blob.core.windows.net/input/gmw/gmw_v3_2020_vec.shp /code/data/
ADD https://deppcpublicstorage.blob.core.windows.net/input/gmw/gmw_v3_2020_vec.dbf /code/data/
ADD https://deppcpublicstorage.blob.core.windows.net/input/gmw/gmw_v3_2020_vec.shx /code/data/
ADD https://deppcpublicstorage.blob.core.windows.net/input/gmw/gmw_v3_2020_vec.prj /code/data/

WORKDIR /code

# Don't use old pygeos
ENV USE_PYGEOS=0
