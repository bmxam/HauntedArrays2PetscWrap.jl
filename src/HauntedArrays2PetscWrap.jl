module HauntedArrays2PetscWrap
using HauntedArrays
using PetscWrap
using MPI
using LinearAlgebra
using MPIUtils
using SciMLBase
using DiffEqBase
using LinearSolve
using SparseArrays

function DiffEqBase.recursive_length(A::HauntedVector)
    MPI.Allreduce(n_own_rows(A), MPI.SUM, get_comm(A))
end

const init_petsc = PetscInitialize
const finalize_petsc = PetscFinalize
export init_petsc, finalize_petsc

include("utils.jl")

include("cache.jl")
export PetscCache

include("algebra.jl")
include("solvers.jl")
export PetscFactorization

end
