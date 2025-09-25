===============
Qbiome
===============

.. class:: no-web no-pdf

:Info: Draft link will be posted here
:Author: ZeD Lab <zed.createuky.net>
:Description: Robust Odds Ratio correcting label noise 
:Documentation: 


**Usage:**

.. code-block::

    from lambda_or import lambda_or, pq_from_two_gates

    tilde = np.array([[100, 50],[ 80, 70]], dtype=float)
    # Suppose selection-conditional rates from two-gate ROC:
    p_sel, q_sel = 0.92, 0.90
    res = lambda_or(tilde, p_sel, q_sel, n_val=2000)
    
    print(res.log_or, res.neglog10_p, res.z, res.se)
    print(res.counts)   # corrected a,b,c,d

