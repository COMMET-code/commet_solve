Example 1: running a simulation without an NCM
==============================================

This first example is designed to give users a simple first exposure to the input file interface -- it can be thought of as a ''hello world'' equivalent.
The learning outcomes are as follows:

* First exposure to the COMMET input file schema
* Using built-in meshes
* Defining traditional (not NCM) hyperelastic material models 
* Defining Dirichlet boundary conditions
* Choosing field outputs

Example files 
-------------

The files for running the example can be downloaded as a single zip file `here <../../_static/example_1.zip>`_
and it contains the following files:

| .
| ├── no-ncm.jsonc
| └── visualize.psvm

Problem definition
------------------
We simulate a square plate with a hole in its centre and utilize quarter-symmetry. 
The problem is displayed schematically in the figure below.

.. TODO: Change fig so that doesn't have current configuration and applied displacement is shown
.. image:: geom.png
   :width: 300

The plate consists of a Neohookean material with the following strain energy density function

.. math::

   \Psi(\mathbf{C}) = \frac{\mu}{2}\left[I_1 - 3 - \log(I_3)\right] + \frac{\lambda}{4}\left[I_3 -1 -\log(I_3)\right] \,.

Here, :math:`\mu` and :math:`\lambda` are material parameters, and :math:`I_1=\text{tr}(\mathbf{C})` and :math:`I_3=\text{det}(\mathbf{C})`.

The input file
--------------

The input files for COMMET are discussed in detail in :ref:`input-files`, however we repeat some of the keypoints here.
COMMET uses JSON (or JSON with comments) for input files -- the file extension "\*.json" denotes a standard JSON file whereas "\*.jsonc" denotes JSON with comments.
In each input file, there are several sections that may be defined.
The sections that will be defined in this example are `mesh`, `materials`, `stages`, and `outputs`.


The input file to define the problem is shown below. 

.. code-block:: json
   :caption: Content of no-ncm.jsonc


    {
        "mesh": {
            "from": "built-in",
            "type": "quarter_plate_with_hole",
            "order": 1,
            "radius": 0.2,
            "length": 1,
            "thickness": 0.1,
            "planar_refinements": 3,
            "global_refinements": 1
        },
        "materials": [
            {
                "id": 0,
                "type": "traditional",
                "elasticity": {
                    "type": "neohookean",
                    "mu": 77,
                    "lambda": 115
                }
            }
        ],
        "stages": [
            {
                "end_time": 1,
                "time_increment_size": 0.1,
                "dirichlet_boundary_conditions": [
                    {
                        "type": "standard",
                        "boundary_id": 1,
                        "components": [
                            0
                        ],
                        "values": [
                            0
                        ]
                    },
                    {
                        "type": "standard",
                        "boundary_id": 4,
                        "components": [
                            0,
                            1,
                            2
                        ],
                        "values": [
                            1,
                            0,
                            0
                        ]
                    }
                ],
                "neumann_boundary_conditions": [],
                "robin_boundary_conditions": []
            }
        ], 
        "outputs": {
             "scalar_outputs": ["I1"],
             "vector_outputs": [ ],
             "tensor_outputs": ["kirchhoff_stress", "C"]
         },
        "miscellaneous": {
            "output_vtus": true,
            "nr_threshold": 1e-6,
            "max_nr_iterations": 5
        }
    }


We'll break each section down in what follows.

.. However, blocks in the input files tend to have a pattern where the "type" or "class" of a block is specified by the key ``type``, and then the parameters required for instantiating the object associated with that class in the solver are provided by other keys.

.. However, not all 

Mesh definition
^^^^^^^^^^^^^^^
The mesh for the problem is defined in the following block.

.. code-block:: json

    "mesh": {
            "order": 1,
            "from": "built-in",
            "type": "quarter_plate_with_hole",
            "radius": 0.2,
            "length": 1,
            "thickness": 0.1,
            "planar_refinements": 3,
            "global_refinements": 1
    }

For each COMMET simulation, the ``mesh`` section must be defined (some sections in the input file are optional, but the mesh section is not).
The block has the following keys:

* ``order`` (optional, default is 1) the order of the finite element to use.
* ``from`` (required) may either be set to ``"read"``  to be read from a file or ``"built-in"`` to use one of the built-in meshes.
* ``type`` (required if ``"from"="built-in"``) If ``from`` is set to ``"built-in"`` then you must specify which built-in mesh to use here. See :ref:`built-in-meshes` for options.
* ``radius`` (optional) Radius of the hole in the plate
* ``length`` (optional) Length of the plate
* ``thickness`` (optional) Thickness of the plate
* ``planar_refinements`` (optional) The number of times to refine the mesh of the geometry in-plane
* ``global_refines`` (optional) The number of times to refine the mesh of the geometry globally (through the thickness and in-plane)


Material definition
^^^^^^^^^^^^^^^^^^^

The materials used in the problem are defined in the following block.

.. code-block:: json

    "materials": [
         {
                "id": 0,
                "type": "traditional",
                "elasticity": {
                    "type": "neohookean",
                    "mu": 77,
                    "lambda": 115
                }
         }
    ]


A mesh may contain multiple materials.
As such, the materials used in the problem are defined in a list.
For this problem, however, there is only one material and so the list only has one entry



Stage definition
^^^^^^^^^^^^^^^^


Output definition
^^^^^^^^^^^^^^^^^

Running the simulation
----------------------

.. code-block:: console
   :caption: Running the simulation

   $ mpirun -np $NP commet_solve no-ncm.jsonc


Results
-------

