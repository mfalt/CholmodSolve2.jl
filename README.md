# CholmodSolve2

[![Build Status](https://travis-ci.org/mfalt/CholmodSolve2.jl.svg?branch=master)](https://travis-ci.org/mfalt/CholmodSolve2.jl)

[![Coverage Status](https://coveralls.io/repos/mfalt/CholmodSolve2.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/mfalt/CholmodSolve2.jl?branch=master)

[![codecov.io](http://codecov.io/github/mfalt/CholmodSolve2.jl/coverage.svg?branch=master)](http://codecov.io/github/mfalt/CholmodSolve2.jl?branch=master)

Package for solving linear systems given an LDLt factorization.

This package supplies a wrapper for the solve2 routine in SuiteSparse/CHOLMOD, callable using `A_ldiv_B!(c, F, b)`. The package keeps the necessary workspace variables in memory to avoid new allocations on every solve.

Example:
```julia
using CholmodSolve2
m = 400;  n = 500;
A = randn(m, n);
Q = sparse([I A'; A -I]);
x = randn(m+n); y = similar(x);
F = ldltfact(Q)
A_ldiv_B!(y, F, x) # Will do some allocations
A_ldiv_B!(y, F, x) # Should be free of allocations
Q*y ≈ x
```
CHOLMOD reallocates workspace variables if they do not have correct dimensions, even if the allocated space is large enough. This package tries to avoid this by manually reshaping the workspace variables before sending them to CHOLMOD.    

There is currently no support for complex vectors or sparse RHS.
