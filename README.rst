===============
Lambda-OR
===============

.. image:: https://zenodo.org/badge/1063683292.svg
  :target: https://doi.org/10.5281/zenodo.17196710


.. class:: no-web no-pdf

:Info: Draft link will be posted here
:Author: ZeD Lab <zed.createuky.net>
:Description: Robust Odds Ratio correcting label noise 
:Documentation: 


**Usage:**

.. code-block::

    from lambda_or import lambda_or, pq_from_two_gates
    from lambda_or import lambda_or, neglog10_p_from_z
    tilde = np.array([[100, 50],[ 80, 70]], dtype=float)

    # lambda-OR
    res = lambda_or(
    tilde_counts=tilde,
    p_sel=0.92,
    q_sel=0.88,
    n_val=1000
    )
