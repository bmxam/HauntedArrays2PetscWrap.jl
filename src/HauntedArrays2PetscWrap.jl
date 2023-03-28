module HauntedArrays2PetscWrap
using HauntedArrays
using PetscWrap

include("cache.jl")
export PetscCache

include("convert.jl")
export get_updated_petsc_array

end