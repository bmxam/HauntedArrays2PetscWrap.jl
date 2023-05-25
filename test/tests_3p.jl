module test
using MPI
using PetscWrap
using HauntedArrays
using HauntedArrays2PetscWrap
using Random
using MPIUtils

rng = MersenneTwister(1234)

MPI.Initialized() || MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
np = MPI.Comm_size(comm)
mypart = rank + 1

@assert np == 3 "Tests to be ran on three procs only"

function test1()
    nx = 2
    lid2gid, lid2part = HauntedArrays.generate_1d_partitioning(nx, mypart, np, true, rng)

    x = HauntedVector(comm, lid2gid, lid2part; cacheType = PetscCache)
    x .= rand(length(lid2gid))

    _x = HauntedArrays2PetscWrap.get_updated_petsc_array(x)
    y = vec2array(_x)

    @one_at_a_time begin
        @show owned_values(x) == y
    end
end

test1()

end