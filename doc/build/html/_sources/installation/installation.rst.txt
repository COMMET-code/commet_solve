Installation
============


.. _Docker engine install: https://docs.docker.com/engine/install/

.. _Docker desktop install: https://docs.docker.com/desktop/setup/install/windows-install/

.. _Docker hub: https://hub.docker.com/repository/docker/benalheit/commet/general

.. _COMMET Github repo: https://github.com/COMMET-code/commet_solve

For running COMMET on a laptop or workstation we recommend the Docker installation route as this is simplest.

If you would like to edit the underlying code though you would obviously need to compile from source.

Additionally, for running COMMET on an HPC with distributed compute nodes you would likely need to also compile from source. 

Docker
------

First install docker by following the directions at `Docker engine install`_ or `Docker desktop install`_.

Get the image
^^^^^^^^^^^^^

From the terminal, you will be able to pull the docker image from `Docker hub`_ using the following command:

.. code-block:: console

   $ docker pull benalheit/commet

That's it, you now have COMMET installed. 

Verify that its working
^^^^^^^^^^^^^^^^^^^^^^^
.. TODO: Link to example here
To verify that the installation is working, you can download an example and run it.
To run an example, open a terminal in a directory where the files for the example are located.
Then run the following command:

.. code-block:: console

   $ docker run --rm -v ./:/data -w /data benalheit/commet mpirun -n <n_procs_to_use> commet_solve <example_input_file>

This slightly long command starts up a docker container with the following settings:

* :code:`--rm` removes the container after it finishes running -- this keeps things tidy.
* :code:`-v ./:data` mounts the current directory on your machine :code:`./` to a directory in the container :code:`/data`
* :code:`-w /data` sets the working directory inside the container to :code:`/data`
* :code:`benalheit/commet` is the image from which Docker will create the container
* :code:`mpirun -n 2 commet_solve <example_input_file>` is the command that will be run once the docker container has started up (and has changed directory to :code:`/data`) 


From source (Linux/WSL)
-----------------------

Building COMMET solve from source is a long and arduous process because it contains several large dependencies.
As such, we strongly recommend that users use the docker image instead. 
The primary use cases for building from source are:

1. Needing to use COMMET on an HPC cluster (which typically doesn't work with docker).
2. Needing to make changes to the COMMET code base.

In these cases, the instructions for building the code can obtained from the `COMMET Github repo`_.

