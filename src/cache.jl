struct PetscCache{P,I} <: HauntedArrays.AbstractCache
    # Petsc.Vec, Petsc.Mat or Nothing
    array::P

    # Local to PETSc indexing
    lid2pid::Vector{I}

    # Total number of rows (= number of cols if matrix)
    nrows_glob::I

    PetscCache(a, l2p::Vector{I}, n::I) where {I<:Integer} = new{typeof(a),I}(a, l2p, n)
end

function HauntedArrays.build_cache(
    ::Type{<:PetscCache},
    array::AbstractArray{T,N},
    exchanger::HauntedArrays.AbstractExchanger,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    oid2lid::Vector{I}
) where {T,N,I<:Integer}
    # Alias
    comm = get_comm(exchanger)

    # Number of elements handled by each proc
    n_by_rank = MPI.Allgather(length(oid2lid), comm)
    nrows_glob = sum(n_by_rank)

    # Init Petsc if needed (should use options here)
    PetscInitialized() || PetscInitialize()

    # Compute local to petsc numbering
    lid2pid = _compute_lid2pid(exchanger, n_by_rank, lid2gid, oid2lid)

    array = _build_petsc_array(comm, length(oid2lid), N)

    return PetscCache(array, lid2pid, nrows_glob)
end

"""
local id to petsc id
"""
function _compute_lid2pid(
    exchanger::HauntedArrays.AbstractExchanger,
    n_by_rank,
    lid2gid::Vector{I},
    oid2lid::Vector{I},
) where {I<:Integer}
    # Alias
    rank = MPI.Comm_rank(get_comm(exchanger))

    # Compute the offset by calculating the number of elements
    # handled by procs with a lower rank
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
        # todo...
        array = nothing
    end
    return array
end

function HauntedArrays.copy_cache(cache::PetscCache)
    array = duplicate(cache.array)
    return PetscCache(array, cache.lid2pid, cache.nrows_glob)
end

PetscWrap.duplicate(::Nothing) = nothing