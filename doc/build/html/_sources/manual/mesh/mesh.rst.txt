Mesh
====


.. toctree::
   :maxdepth: 0
   :hidden: 

   builtin
   read


.. admonition:: ``mesh``

   **Parameters**

   * ``from`` (enum) **required**: whether to read in a mesh or use a built-in mesh (``"read"`` or ``"built-in"``).
   * ``path`` (string) **required if ``from=read``**: path to mesh that will be read in.
   * ``type`` (enum) **required if ``from=builtin``**: type of mesh to generate (see :ref:`built-in-meshes` for options)
   * ``order`` (int) **required**: Order of the finite element to use.


   **Examples** 

   .. code-block:: json

      "mesh": {
          "from": "read",
          "path": "./my-mesh.msh",
          "order": 1,
          "global_refines": 0
      }


   .. code-block:: json

      "mesh": {
          "from": "built-in",
          "type": "cube",
          "order": 1,
          "global_refines": 4
      }


