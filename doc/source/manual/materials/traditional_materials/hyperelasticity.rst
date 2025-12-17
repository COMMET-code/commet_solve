.. _hyperelasticity:
Hyperelasticity
===============

.. toctree::
   :maxdepth: 2
   :hidden: 

   standard_materials
   iso_vol_materials


.. admonition:: ``Elasticity material specification`` 

   **Parameters**
       * ``id`` (int) **required**: ID of material domain.
       * ``orientation_vectors`` (list[string]) **optional**: List of field names to be used as orientation vectors (see :ref:`fields`)
       * ``type`` (enum) **required**: 
           * options: ``"ncm"`` | ``"traditional"`` (see :ref:`materials`)
           * for this extension: ``"type"="traditional"`` 
       * ``elasticity`` (json) **required**: description of the elastic material behaviour (see :ref:`hyperelasticity`).


   **Examples** 

   .. code-block:: json

      "materials": [
      { 
          "id": 0,
          "type": "traditional",
          "elasticity": {
              // specification for hyperelastic material behaviour
          }
      },
      { 
          "id": 1,
          "type": "traditional",
          "orientation_vectors": ["my fibre field"],
          "elasticity": {
              // specification for hyperelastic material behaviour
          }
      }
      ]

