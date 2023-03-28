module test
using MPI
using PetscWrap
using HauntedArrays
using HauntedArrays2PetscWrap
using Random

rng = MersenneTwister(1234)

MPI.Initialized() || MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
np = MPI.Comm_size(comm)

@assert np == 1 "Tests to be ran on one proc only"

function test1()
    lid2gid = [6, 5, 4, 3, 2, 1]
    lid2part = ones(Int, length(lid2gid))
    x = HauntedVector(comm, lid2gid, lid2part; cacheType = PetscCache)
    x .= [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

    _x = get_updated_petsc_array(x)

    y = vec2array(_x)

    @show x == y
end

function test_ldiv()
    lid2gid = [6, 5, 4, 3, 2, 1]
    n = length(lid2gid)
    lid2part = ones(Int, n)
    b = HauntedVector(comm, lid2gid, lid2part; cacheType = PetscCache)
    b .= rand(rng, n)

    A = similar(b, n, n)
    A .= rand(rng, n, n)

    x = A \ b
    _x = parent(A) \ parent(b)
    for rtol in [1e-20 * 10^n for n = 0:10]
        @show rtol, all(isapprox.(_x, parent(x); rtol = rtol))
    end
end

test1()
test_ldiv()

isinteractive() || MPI.Finalize()
end