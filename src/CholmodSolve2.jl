module CholmodSolve2

import Base.SparseArrays.CHOLMOD: SuiteSparse_long, Factor,
        C_Dense, C_Sparse, C_Factor, @cholmod_name, Cint, CHOLMOD_A, common,
        CHOLMODException, xtyp

global const DEBUG = false
# Only allow Float64 for now
global const Tv = Float64 # TODO Relax to VTypes

# Julia, dont CG these, or we will have problems!!
CHOLMOD_Bp = Ptr{C_Dense{Tv}}(C_NULL)
CHOLMOD_Xp = Ptr{C_Dense{Tv}}(C_NULL)
#CHOLMOD_Xsetp = Ptr{C_Sparse{Tv}}(C_NULL)
CHOLMOD_Yp = Ptr{C_Dense{Tv}}(C_NULL)
CHOLMOD_Ep = Ptr{C_Dense{Tv}}(C_NULL)

# Workspace pointers that CHOLMOD will use. CHOLMOD will take care of ALL allocations
global const CHOLMOD_BHandle     = convert(Ptr{Ptr{C_Dense{Tv}}},pointer_from_objref(CHOLMOD_Bp))
global const CHOLMOD_XHandle     = convert(Ptr{Ptr{C_Dense{Tv}}},pointer_from_objref(CHOLMOD_Xp))
#global const CHOLMOD_Xset_Handle = convert(Ptr{Ptr{C_Sparse{Tv}}},pointer_from_objref(CHOLMOD_Xsetp))
global const CHOLMOD_YHandle     = convert(Ptr{Ptr{C_Dense{Tv}}},pointer_from_objref(CHOLMOD_Yp))
global const CHOLMOD_EHandle     = convert(Ptr{Ptr{C_Dense{Tv}}},pointer_from_objref(CHOLMOD_Ep))

function solve2!(sys::Integer, F::Factor{Tv}, B::Ptr{C_Dense{Tv}})
    if size(F,1) != unsafe_load(B).nrow
        throw(DimensionMismatch("B must have as many rows as F in A_ldiv_B!"))
    end

    cmn = common()
    d = ccall((@cholmod_name("solve2", SuiteSparse_long),:libcholmod), Cint,
            (Cint, Ptr{C_Factor{Tv}},       # sys, *L
            Ptr{C_Dense{Tv}},   Ptr{Void},  # *B  *Bset
            Ptr{Void},          Ptr{Void},  # **X **Xset
            Ptr{Void},          Ptr{Void},  # **Y **E
            Ref{UInt8}),                    # *common
            sys, get(F.p),
            B, C_NULL,                      # Bset = C_NULL
            CHOLMOD_XHandle, C_NULL,        # Xset = C_NULL
            CHOLMOD_YHandle, CHOLMOD_EHandle,
            cmn)
    if CHOLMOD_XHandle == C_NULL
        throw(CHOLMODException("Solve failed with unexpected error"))
    elseif unsafe_load(CHOLMOD_XHandle) == C_NULL
        throw(CHOLMODException("Solve failed with NULL output"))
    end
    return d
end

function Base.A_ldiv_B!(C::AbstractVecOrMat{Tv}, F::Factor{Tv}, B::AbstractVecOrMat{Tv}) where Tv
    if size(B,1) != size(F,1) || size(C,1) != size(F,1)
        throw(DimensionMismatch("B and C must have same number of rows as F, got size(F)=$(size(F)), size(B)=$(size(B)), size(C)=$(size(C))."))
    end
    if size(C,2) != size(B,2)
        throw(DimensionMismatch("B and C must have same number of columns, got size(B)=$(size(B)), size(C)=$(size(C))."))
    end
    # Allocate Dense B if needed
    DEBUG && println("Julia ensure B: : $(size(B,1)), $(size(B,2))")
    ensure_dense(CHOLMOD_BHandle, size(B,1), size(B,2), size(B,1))
    # Copy B to CHOLMOD input
    _copy!(unsafe_load(CHOLMOD_BHandle), B)
    #Try to resize Y to avoid unnessesary allocation in solve2
    if CHOLMOD_YHandle != C_NULL && unsafe_load(CHOLMOD_YHandle) != C_NULL
        Yp = unsafe_load(CHOLMOD_YHandle)
        Y = unsafe_load(Yp)
        # This is how many rows solve2 wants in Y
        newrows = max(4, size(B,2))
        # TODO We don't dare to change ncol yet
        if Y.ncol == size(B,1) && Y.nrow == Y.d && Y.nzmax >= Y.ncol*newrows
            # Do change nrow and d to what solve2 wants
            DEBUG && println("Julia ensure Y: $newrows, $(Y.ncol)")
            ensure_dense(CHOLMOD_YHandle, newrows, Y.ncol, newrows)
        end
    end
    # Only needed when dimensions of B actually change, but solve2 won't handle it nicely
    DEBUG && println("Julia ensure X: : $(size(B,1)), $(size(B,2))")
    ensure_dense(CHOLMOD_XHandle, size(B,1), size(B,2), size(B,1))

    status = solve2!(CHOLMOD_A, F, unsafe_load(CHOLMOD_BHandle))
    if status != 1
        throw(DimensionMismatch("Solve failed"))
    end
    # Copy the CHOLMOD output to C
    _copy!(C, unsafe_load(CHOLMOD_XHandle))

    return C
end

function ensure_dense(XHandle::Ptr{Ptr{C_Dense{Tv}}}, nrow, ncol, d)
    if XHandle != C_NULL
        XDensep = unsafe_load(XHandle)
        if XDensep != C_NULL
            XDense = unsafe_load(XDensep)
            # To be safe, only do if nrow=ncol
            if XDense.xtype == xtyp(Tv) && nrow == d && XDense.nzmax >= d*ncol
                #We can just change the dimensions, by owerwringing memory we are not allowed to write to...
                DEBUG && println("Julia changed dimensions")
                unsafe_store!(Base.convert(Ptr{Csize_t}, XDensep), nrow, 1) # nrow
                unsafe_store!(Base.convert(Ptr{Csize_t}, XDensep), ncol, 2)     # ncol
                unsafe_store!(Base.convert(Ptr{Csize_t}, XDensep), nrow, 4) # d
                return
            end
        end
    end
    DEBUG && println("Julia couldn't change dimensions")
    X = ccall((@cholmod_name("ensure_dense", SuiteSparse_long),:libcholmod), Ptr{C_Dense{Tv}},
           (Ptr{Ptr{C_Dense{Tv}}},
            Csize_t, Csize_t, Csize_t, Cint, Ptr{UInt8}),
            XHandle,
            nrow, ncol, d, xtyp(Tv), common())
    if XHandle == C_NULL
        throw(CHOLMODException("Ensure dense failed with NULL pointer, this should not happen"))
    elseif unsafe_load(XHandle) == C_NULL
        throw(CHOLMODException("Ensure dense failed in CHOLMOD"))
    end
    return
end

function _copy!(dest::AbstractArray{Tv}, D::Ptr{C_Dense{Tv}})
    s = unsafe_load(D)
    n = s.nrow*s.ncol
    n <= length(dest) || throw(BoundsError(dest, n))
    if s.d == s.nrow && isa(dest, Array)
        unsafe_copy!(pointer(dest), s.x, s.d*s.ncol)
    else
        k = 0
        for j = 1:s.ncol
            for i = 1:s.nrow
                dest[k+=1] = unsafe_load(s.x, i + (j - 1)*s.d)
            end
        end
    end
    dest
end

# TODO verify correctness when s.d != s.nrow
function _copy!(D::Ptr{C_Dense{Tv}}, orig::AbstractVecOrMat{Tv})
    s = unsafe_load(D)
    n = s.nrow*s.ncol
    length(orig) <= n || throw(BoundsError(n, orig))
    if s.d == s.nrow && isa(orig, Array)
        unsafe_copy!(s.x, pointer(orig), length(orig))
    else
        k = 0
        for j = 1:s.ncol
            for i = 1:s.nrow
                unsafe_store!(s.x, orig[k+=1], i + (j - 1)*s.d)
            end
        end
    end
    D
end

# For debugging memory use
function print_common()
    Base.SparseArrays.CHOLMOD.set_print_level(common(), 5)
    ccall((@cholmod_name("print_common", SuiteSparse_long),:libcholmod), Cint,
                (Ptr{UInt8}, Ref{UInt8}),                                    # *common
                    "Test1", common())
end

end # module
