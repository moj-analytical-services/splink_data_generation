FROM jupyter/all-spark-notebook:8882c505faa8

COPY . ${HOME}
USER root



RUN pip install splink
RUN pip install .
RUN pip install altair
RUN pip install altair_data_server


RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

