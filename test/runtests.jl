using CholmodSolve2
using Base.Test

m = 400;  n = 500;
A = randn(m, n);
Q = sparse([I A'; A -I]);
y = randn(m+n); x = randn(m+n);
F = ldltfact(Q)

val, t, bytes, gctime, memallocs = @timed A_ldiv_B!(x, F, y)
@test bytes > 500000# should be roughly 1432298
@test Q*x ≈ y
# Solve again, without allocs
val, t, bytes, gctime, memallocs = @timed A_ldiv_B!(x, F, y)
@test bytes < 400 # should be 192
@test Q*x ≈ y

# Compile for array of dimension 2 (still same size)
y2 = Array{Float64,2}(m+n,1); y2 .= x
x2 = Array{Float64,2}(m+n,1); x2 .= x
val, t, bytes, gctime, memallocs = @timed A_ldiv_B!(x2, F, y2)

y2 = [y 2y]
x2 = [x 2x]
# Test allocation with double size
# We will probably not allocate too much here (only X_Handle, and B_Handle)
val, t, bytes, gctime, memallocs = @timed A_ldiv_B!(x2, F, y2)
@test bytes < 30000# should be roughly 900*2*2*8=28800
@test Q*x2 ≈ y2

# Go back to 1 dimension and make sure no allocations
val, t, bytes, gctime, memallocs = @timed A_ldiv_B!(x, F, y)
@test bytes < 400 # should be 192
@test Q*x ≈ y

# Try with many dimensions (since CHOLMOD solves with 4 columns at a time)
y2 = [y 2y 3y 4y]
x2 = [x 2x 3x 4x]
# We will probably not allocate too much here (only X_Handle, and B_Handle)
val, t, bytes, gctime, memallocs = @timed A_ldiv_B!(x2, F, y2)
@test bytes < 60000 # should be roughly 900*2*4*8=57600
@test Q*x2 ≈ y2

y2 = [y2 y2]
x2 = [x2 x2]
# We also allocate new Y_Handle (not only X_Handle, and B_Handle)
val, t, bytes, gctime, memallocs = @timed A_ldiv_B!(x2, F, y2)
@test bytes < 180000 # should be roughly 900*3*8*8=172800
@test Q*x2 ≈ y2

# Change size of F
m = 200;  n = 300;
A2 = randn(m, n);
Q2 = sparse([I A2'; A2 -I]);
y3 = randn(m+n); x3 = randn(m+n);
F2 = ldltfact(Q2)

val, t, bytes, gctime, memallocs = @timed A_ldiv_B!(x3, F2, y3)
# We don't update size of Y here, so CHOLMOD will reallocate it
#@test bytes < 400
@test Q2*x3 ≈ y3

# Just test that print_common doesn't fail
ok = CholmodSolve2.print_common()
@test ok == 1
# Expected result:
# memory blocks in use:          39
# memory in use (MB):           8.9
# peak memory usage (MB):      16.6
