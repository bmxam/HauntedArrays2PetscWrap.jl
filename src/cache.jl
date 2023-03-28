struct PetscCache{P,I} <: HauntedArrays.AbstractCache where {I<:Integer}
    # Petsc.Vec, Petsc.Mat or Nothing
    array::P

    # Local to PETSc indexing
    lid2pid::Vector{I}

    PetscCache(array, l2p::Vector{I}) where {I<:Integer} = new{typeof(array),I}(a, l2p)
end

function HauntedArrays.build_cache(
    ::Type{PetscCache},
    exchanger::AbstractExchanger,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    oid2lid::Vector{I},
    ndims::Int,
    T,
) where {I<:Integer}
    comm = get_comm(exchanger)

    # Compute local to petsc numbering
    lid2pid = _compute_lid2pid(exchanger, lid2gid, oid2lid)

    array = _build_petsc_array(comm, length(oid2lid), ndims)

    return PetscCache(array, lid2pid)
end

"""
local id to petsc id
"""
function _compute_lid2pid(
    exchanger::HauntedArrays.AbstractExchanger,
    lid2gid::Vector{I},
    oid2lid::Vector{I},
) where {I<:Integer}
    # Alias
    comm = get_comm(exchanger)
    rank = MPI.Comm_rank(comm)

    # Compute the offset by calculating the number of elements
    # handled by procs with a lower rank
    n_by_rank = MPI.Allgather(length(oid2lid), comm)
    offset = sum(n_by_rank[1:rank])

    # Create an Array whose values will be
    # the Petsc id of each element (including ghosts)
    lid2pid = similar(lid2gid)
    lid2pid[oid2lid] .= collect(1:length(oid2lid)) .+ offset
    update_ghosts!(lid2pid, exchanger)

    return lid2pid
end

function _build_petsc_array(comm::MPI.Comm, n_own_rows::Int, ndims::Int)
    # Allocate PETSc array
    if ndims == 1
        array = create_vector(comm; nrows_loc = n_own_rows, autosetup = true)
    else
        array = nothing
    end
    return array
end

function copy_cache(cache::PetscCache)
    array = duplicate(cache.array)
    return PetscCache(array, cache.lid2pid)
end

PetscWrap.duplicate(::Nothing) = nothing