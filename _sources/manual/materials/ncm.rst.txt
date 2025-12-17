.. _ncm:
NCM
===



.. admonition:: ``NCM material specification`` 

   **Description**
       Extension of ``Base MaterialSpec`` (see :ref:`materials`) for NCMs.

   **Parameters**
       * ``id`` (int) **required**: ID of material domain.
       * ``orientation_vectors`` (list[string]) **optional**: List of field names to be used as orientation vectors (see :ref:`fields`)
       * ``type`` (enum) **required**: 
           * options: ``"ncm"`` | ``"traditional"`` (see :ref:`materials`)
           * for this extension: ``"type"="ncm"`` 
       * ``path_to_torchscript`` (string) **required**: path to the torchscript file containing the NCM.
       * ``evaluation_method`` (enum) **required**: ``using_F`` | ``using_C`` | ``optimized``
       * ``vectorization`` (enum) **required**: ``global`` | ``batched``
       * ``batch_size`` (int) **required if "vectorization"=="batched"**: number of material points per batch.


   **Examples** 

   .. code-block:: json

      "materials": [
      { 
          "id": 0,
          "type": "ncm",
          "path_to_torchscript": "./ickan.torchscript",
          "evaluation_method": "using_F",
          "vectorization": "global"
      },
      { 
          "id": 1,
          "type": "ncm",
          "path_to_torchscript": "./micnn.torchscript",
          "evaluation_method": "optimized", 
          "vectorization": "batched", 
          "batch_size": 512
      }
      ]

