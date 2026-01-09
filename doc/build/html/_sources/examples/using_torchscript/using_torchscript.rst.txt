.. _defining material models:
Defining material models using Pytorch
======================================

.. _Torchscript docs: https://docs.pytorch.org/docs/2.8/jit.html

.. _Creating an NN in using Pytorch: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

.. _COMMET paper: https://arxiv.org/abs/2510.00884

COMMET allows for the use of material models implemented using pytorch.
This is done by using *torchscript* (see `Torchscript docs`_).

The learning outcomes are as follows:

* How to define a model using Pytorch that can later be used in a COMMET simulation.
* How to export a Pytorch model using torchscript.
* How to read a torchscript file into a commet simulation.

Example files 
-------------

The files for running the example can be downloaded as a single zip file `here <../../_static/using_torchscript.zip>`_
and it contains the following files:

| .                           
| ├── nh.torchscript          
| ├── nh_torchscript_model.py   
| └── using_torchscript.jsonc 

Problem definition
------------------

The problem definition is identical to the Hello World example (see :ref:`helloworld`).
However, we will implement the Neohookean material model in Python using Pytorch, export the model as torchscript, and read in the model in the COMMET solver.

Defining a material model using Pytorch
---------------------------------------

We assume that a reader has some prior experience with Pytorch, in particular, one needs to be able to construct neural networks using Pytorch (see, e.g. `Creating an NN in using Pytorch`_).
To implement a material model with Pytorch for later use in COMMET one must write a class that inherits from :code:`torch.nn.Module`.
That class will then be serialized into a torchscript file which will later be read into COMMET.
When the model is read in, COMMET will search for one of several possible methods of the class which ultimately define the material behaviour -- it is important that one of these methods is defined otherwise COMMET will not know where to look for the definition of the material behaviour.
These methods, with their signatures, are as displayed in the following code listing.

.. code-block:: python
   :linenos:

    import torch
    from torch import nn

    class YourMaterialModel(nn.Module):
        def __init__(self, ...):
            ...

        @torch.jit.export
        def W_NN_from_C(self,
                        C: torch.Tensor,
                        structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Returns the strain energy density for a given right Cauchy-Green tensor and structural vectors.
            ***Important***: COMMET takes advantage of symmetries in the right Cauchy-Green tensor when applying AD to calculate the stress and stiffness.
            In order for this to work correctly, it is important that the first line of this method is:
                C = 0.5*(C + C.transpose(1, 2)) 


            Parameters
            ----------
            C : torch.Tensor
                **shape (batch size, 3, 3)** Right Cauchy-Green tensors.
            structural_vectors : torch.Tensor
                **shape (batch size, number of structural vectors, 3)**

            Returns
            -------
            torch.Tensor
                **shape (batch size)** strain energy density

            """
            C = 0.5*(C + C.transpose(1, 2)) # !!! This is required for the autograd to pick-up the symmetries correctly !!!
            ...

        @torch.jit.export
        def W_NN_from_F(self,
                        F: torch.Tensor,
                        structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Returns the strain energy density for a given deformation gradient and structural vectors.

            Parameters
            ----------
            F : torch.Tensor
                **shape (batch size, 3, 3)** Deformation gradients.
            structural_vectors : torch.Tensor
                **shape (batch size, number of structural vectors, 3)**

            Returns
            -------
            torch.Tensor
                **shape (batch size)** strain energy density

            """
            ...

        @torch.jit.export
        def psi_tau_cc_from_F(self,
                              F: torch.Tensor,
                              structural_tensors: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """Returns the strain energy density, Kirchhoff stress, and spatial stiffness tensor in Voigt notation for a given deformation gradient and structural vectors.
                The Voigt notation ordering is as follows (zero indexed):
                vgt = [
                    (0, 0),
                    (1, 1),
                    (2, 2),
                    (1, 2),
                    (0, 2),
                    (0, 1)
                    ]

                Parameters
                ----------
                F : torch.Tensor
                    **shape (batch size, 3, 3)** Deformation gradients.
                structural_vectors : torch.Tensor
                    **shape (batch size, number of structural vectors, 3)**

                Returns
                -------
                torch.Tensor
                    **shape (batch size)** strain energy density
                torch.Tensor
                    **shape (batch size, 6)** Kirchhoff stress (2 F d\Psi/dC F^T) in Voigt notation 
                torch.Tensor
                    **shape (batch size, 6, 6)** Spatial stiffness tensor (4 d\Psi/(dC_{IJ} dC_{KL} ) F_{iI}F_{jJ}F_{kK}F_{lL}) in Voigt notation 

                """
                ...


You may note that the methods :code:`W_NN_from_F` and :code:`W_NN_from_C` only require calculation of the strain energy density, whereas :code:`psi_tau_cc_from_F` also requires calculation of the stress and stiffness.
This is because, if :code:`W_NN_from_F` or :code:`W_NN_from_C` is used, COMMET will use automatic differentiation (AD) to calculate the stress and stiffness.
In contrast, the method :code:`psi_tau_cc_from_F` allows a user to implement the necessary derivatives themself, which may be more efficient than using AD.
The decorators :code:`@torch.jit.export` are required to indicate that Pytorch must include these methods when generating the torchscript file (we'll show how this is done below).
Finally, note that a user only needs to implement one of these methods, not all of them. 

Below we show a full example of how to implement a simple Neohookean model using Pytorch and export an associated torchscript file -- this is the contents of :code:`nh_torchscript_model.py`.

.. code-block:: python
    :linenos:

    import torch
    from torch import nn
    from typing import Optional


    class NH(nn.Module):
        def __init__(self,
                     mu: float,
                     lam: float):
            super().__init__()
            self.mu = torch.nn.Parameter(torch.Tensor([mu]))
            self.lam = torch.nn.Parameter(torch.Tensor([lam]))

        @torch.jit.export
        def W_NN_from_C(self,
                        C: torch. Tensor,
                        structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Returns the strain energy density for a given right Cauchy-Green tensor and structural vectors.

            Parameters
            ----------
            C : torch.Tensor
                **shape (batch size, 3, 3)** Right Cauchy-Green tensors.
            structural_vectors : torch.Tensor
                **shape (batch size, number of structural vectors, 3)**

            Returns
            -------
            torch.Tensor
                **shape (batch size)** strain energy density

            """

            C = 0.5*(C + C.transpose(1, 2)) # !!! This is required for the autograd to pick-up the symmetries correctly !!!

            I1 = torch.sum(C[:, [0, 1, 2], [0, 1, 2]], dim=1)
            I3 = torch.det(C)
            return 0.5 * self.mu * (I1 - 3 - torch.log(I3)) + 0.25 * self.lam * (I3 - 1 - torch.log(I3))

        @torch.jit.export
        def W_NN_from_F(self,
                        F: torch.Tensor,
                        structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Returns the strain energy density for a given deformation gradient and structural vectors.

            Parameters
            ----------
            F : torch.Tensor
                **shape (batch size, 3, 3)** Deformation gradients.
            structural_vectors : torch.Tensor
                **shape (batch size, number of structural vectors, 3)**

            Returns
            -------
            torch.Tensor
                **shape (batch size)** strain energy density

            """

            C = torch.bmm(F.transpose(1, 2), F)
            return self.W_NN_from_C(C, structural_vectors)

        def forward(self,
                    F: torch.Tensor,
                    structural_vectors: Optional[torch.Tensor] = None):
            return self.W_NN_from_F(F, structural_vectors)


    def main():
        mu: float = 77
        lam: float = 115
        nh: NH = NH(mu, lam)

        dim = 3
        F_mat = torch.zeros(100, dim, dim)

        traced = torch.jit.trace(nh, (F_mat, ))
        traced.save("nh.torchscript")


    if __name__ == "__main__":
        main()


The above example also demonstrates how the model is converted to torchscript, i.e. the model and some example input is provided to :code:`torch.jit.trace`, and the traced output is written to file using the :code:`save` method. 
It is important that the :code:`forward` method is defined for the model since this is the default entry point for the :code:`torch.jit.trace` function.
Next we'll show how to use this in a COMMET simulation.

The input file
--------------

Since the problem definition is identical to the Hello World example (see :ref:`helloworld`), the input file is largely the same.
The only difference is the material definition. 

.. code-block:: json
   :caption: Content of using_torchscript.jsonc
   :linenos: 


    {
        // Defining the mesh 
        "mesh": {
            // Indicating that we will use a built-in mesh (not reading one in from file)
            "from": "built-in", 
            // The type of built-in mesh that we will use
            "type": "quarter_plate_with_hole",
            // The order of the finite elements to use
            "order": 1,
            // Radius of the hole in the plate
            "radius": 0.2,
            // Length of the plate
            "length": 1,
            // Thickness of the plate
            "thickness": 0.1,
            // The number of times to refine the mesh of the geometry in-plane
            "planar_refinements": 1,
            // The number of times to refine the mesh of the geometry globally (through the thickness and in-plane)
            "global_refinements": 1
        },
        // Defining the materials
        "materials": [
            {
                "id": 0,
                // Technically, NCM stands for "neural constitutive network".
                // This just indicates to commet_solve that the model must be loaded from torchscript.
                "type": "ncm", 
                "vectorization": "batched", // batched | global
                "batch_size": 512, // must be defined if "vectorization": "batched"
                "evaluation_method": "using_C", // "optimized" | "using_F" | "using_C"
                "path_to_torchscript": "nh.torchscript"
            }
        ],
        // Defining the boundary conditions to be applied over time
        "stages": [
            {
                "end_time": 1,
                "time_increment_size": 0.1,
                "dirichlet_boundary_conditions": [
                     {
                         "type": "standard", // Either "standard" or "pull_twist"
                         "boundary_id": 1, // The id of the boundary to which this condition applies
                         "components": [0], // The components (0->x, 1->y, 2->z) of the displacement to be prescribed.
                         "values": [0] // The values of the prescribed displacement. Must have the same number of entries as "components"
                     },
                     {
                         // Similar to above
                         "type": "standard",
                         "boundary_id": 2,
                         "components": [1],
                         "values": [0]
                     },
                     {
                         // Similar to above
                         "type": "standard",
                         "boundary_id": 3,
                         "components": [2],
                         "values": [0]
                     },
                     {
                         // Similar to above
                         "type": "standard",
                         "boundary_id": 4,
                         "components": [0, 1, 2],
                         "values": [1, 0, 0]
                     }
                ],
                "neumann_boundary_conditions": [],
                "robin_boundary_conditions": []
            }
        ], 
        // Defining the fields to be output
        "outputs": {
             "scalar_outputs": ["I1"], // Scalar fields to output (here I1=\tr(C))
             "vector_outputs": [ ], // Vector fields to output
             "tensor_outputs": ["kirchhoff_stress", "C" ] // Tensor fields to output
         } 
    }

The details of the of defining the material behaviour are as follows:

* :code:`"type": "ncm"` Setting :code:`type` to code:`"ncm"` indicates to COMMET that the material model must be read in from torchscript
* :code:`"vectorization": "batched"` If the model is read in from torchscript, COMMET uses vectorization to execute constitutive updates for multiple material points simultaneously, i.e. it uses vectorization. A user can choose whether that vectorization is applied globally, i.e. all material points are processed simultaneously, or in batches of a fixed size. In brief, using batched vectorization tends to be faster and more memory efficient than global vectorization. For more details see the `COMMET paper`_.
* :code:`"batch_size": 512` If batch vectorization is chosen over global vectorization then the batch size (i.e. the number of material points to process simultaneously) must also be chosen. The most efficient batch size will likely vary from one problem to another. However, a batch size of 512 or 1024 is typically a good choice. Again, see the `COMMET paper`_ for more details.
* :code:`"evaluation_method": "using_C"` This is where we tell COMMET which method to use when computing the constitutive update:

  * :code:`"using_F"` -> :code:`W_NN_from_F`
  * :code:`"using_C"` -> :code:`W_NN_from_C`
  * :code:`"optimized"` -> :code:`psi_tau_cc_from_F`

* :code:`"path_to_torchscript": "nh.torchscript"` As you might expect, this is just the path to the torchscript file that must be read in.

Running the simulation
----------------------

If you are using docker, you can run the example by going to the directory containing the input file and running

.. code-block:: console

   $ docker run --rm -v ./:/data -w /data commetcode/commet_solve mpirun -n <n_procs_to_use> commet_solve using_torchscript.jsonc

If, instead, you have build COMMET locally and it is in your path, you can run

.. code-block:: console

   $ mpirun -np <n_procs_to_use> commet_solve using_torchscript.jsonc

