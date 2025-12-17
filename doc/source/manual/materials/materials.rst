.. _materials:
Materials
=========


.. toctree::
   :maxdepth: 2
   :hidden: 

   ncm
   traditional


.. admonition:: ``materials`` List[MaterialSpec]


    .. _material_spec:
   **Base MaterialSpec Parameters**

       * ``id`` (int) **required**: ID of material domain.
       * ``type`` (enum) **required**: ``"ncm"`` (see :ref:`ncm`) | ``"traditional"`` (see :ref:`traditional`)
       * ``orientation_vectors`` (list[string]) **optional**: List of field names to be used as orientation vectors (see :ref:`fields`).

   **Examples** 

   .. code-block:: json

      "materials": [
          {
              "id": 0, 
              "orientation_vectors": ["fibre field name"]
              "type": "ncm",
              // ... remaining specification for ncm material type ...
          },
          {
              "id": 1, 
              "type": "traditional",
              // ... remaining specification for traditional material type ...
          }
      ]
      


