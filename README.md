Repacking is a library to test ad-hoc and post-hoc repacking methods to reduce variance for grouping based measurement schemes.

The standard approach to grouping based measurement is to greedily insert strings into at most one group.  These schemes naturally relax these constraints according to simple rules:
In the ad-hoc method, we start with the initial groupings from the sorted insertion algorithm: https://github.com/rmlarose/kcommute/blob/main/kcommute/sorted_insertion.py and then we score all strings according to their expected contribution to the variance c_i^2/N_i.  We then 'repack' strings into the already existing groups in this order according to mutual commutativity.  We pack strings into 1 group at a time and re-score until all possible insertions are exhausted.

In the post-hoc method, we assume the bases are fixed.  We then check to see if any other strings are implicitly diagonalized by the existing bases selections.  If so, they are added to the corresponding group.

Variances are computed according to the theoretical variance formula \sigma_i^2 = (1-<P_i>^2)/N_i

We test on both simulated and hardware data for the H2O Hamiltonian.
