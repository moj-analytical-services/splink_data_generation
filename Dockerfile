FROM jupyter/all-spark-notebook:8882c505faa8

COPY . ${HOME}
USER root



RUN pip install splink
RUN pip install .


RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

