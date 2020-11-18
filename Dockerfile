FROM jupyter/all-spark-notebook:8882c505faa8

COPY . ${HOME}
USER root

RUN git clone https://github.com/moj-analytical-services/splink.git

RUN pip install splink

RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

