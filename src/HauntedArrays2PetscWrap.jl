module HauntedArrays2PetscWrap
using HauntedArrays
using PetscWrap
using MPI
using LinearAlgebra

include("cache.jl")
export PetscCache

include("convert.jl")
export get_updated_petsc_array

include("algebra.jl")

end