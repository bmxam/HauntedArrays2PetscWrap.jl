struct PetscCache{P,I} <: HauntedArrays.AbstractCache
    # Petsc.Vec, Petsc.Mat or Nothing
    array::P

    # Local to PETSc indexing
    lid2pid::Vector{PetscInt}

    # own to PETSc indexing (0-based for petsc)
    oid2pid0::Vector{PetscInt}

    # Use CSR or COO? -> if we keep this, use Trait or type param
    CSR::Bool

    # Index of owned values in I, J, V (only for sparse matrices)
    coo_mask::Vector{I}

    coo_I0::Vector{PetscInt} # 0-based
    coo_J0::Vector{PetscInt} # 0-based

    # Total number of rows (= number of cols if matrix)
    # not needed any more, to be removed
    # nrows_glob::I

    function PetscCache(
        a,
        l2p::Vector{PetscInt},
        o2p::Vector{PetscInt},
        CSR::Bool,
        mask::Vector{I},
        coo_I0::Vector{PetscInt}coo_J0::Vector{PetscInt},
    ) where {I<:Integer}
        new{typeof(a),I}(a, l2p, o2p, CSR, mask, coo_I0, coo_J0)
    end
end

function HauntedArrays.build_cache(
    ::Type{<:PetscCache},
    array::AbstractArray{T,N},
    exchanger::HauntedArrays.AbstractExchanger,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    oid2lid::Vector{I};
    kwargs...,
) where {T,N,I<:Integer}
    # Alias
    comm = get_comm(exchanger)
    my_part = MPI.Comm_rank(comm) + 1

    # Handle kwargs
    CSR = haskey(kwargs, :CSR) ? kwargs[:CSR] : true

    # Number of elements handled by each proc
    n_by_rank = MPI.Allgather(length(oid2lid), comm)
    # nrows_glob = sum(n_by_rank)

    # Init Petsc if needed (should use options here)
    PetscInitialized() || PetscInitialize()

    # Compute local to petsc numbering
    lid2pid = _compute_lid2pid(exchanger, n_by_rank, lid2gid, oid2lid)
    oid2pid0 = lid2gid[oid2lid] .- 1 # allocation is wanted

    # Compute coo mask
    coo_mask = Int[]
    if array isa AbstractSparseMatrix
        _I, _, _ = findnz(array)
        sizehint!(coo_mask, length(_I))
        for (k, li) in enumerate(_I)
            (lid2part[li] == my_part) && push!(coo_mask, k)
        end
    end

    # Build Petsc Array
    _array, coo_I0, coo_J0 =
        _build_petsc_array(comm, array, lid2part, oid2lid, lid2pid, CSR)

    return PetscCache(
        _array,
        PetscInt.(lid2pid),
        PetscInt.(oid2pid0),
        CSR,
        coo_mask,
        coo_I0,
        coo_J0,
    )
end

"""
local id to petsc id. PETSc id is 1-based at this step
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

function _build_petsc_array(
    comm::MPI.Comm,
    ::AbstractVector{T},
    lid2part,
    oid2lid,
    lid2pid,
    ::Bool,
) where {T}
    n_own_rows = length(oid2lid)
    return create_vector(comm; nrows_loc = n_own_rows, autosetup = true),
    PetscInt[],
    PetscInt[]
end

function _build_petsc_array(
    comm::MPI.Comm,
    ::Matrix{T},
    lid2part,
    oid2lid,
    lid2pid,
    ::Bool,
) where {T}
    n_own_rows = length(oid2lid)

    # Allocate PetscMat
    # I don't why I have to set the size like this, but this is the only combination that works
    # TODO : CREATE DENSE MATRIX INSTEAD OF SPARSE WHEN NECESSARY
    return create_matrix(
        comm;
        nrows_loc = n_own_rows,
        ncols_loc = n_own_rows,
        autosetup = true,
    ),
    PetscInt[],
    PetscInt[]
end

function _build_petsc_array(
    comm::MPI.Comm,
    A::AbstractSparseArray{T},
    lid2part,
    oid2lid,
    lid2pid,
    CSR::Bool,
) where {T}
    n_own_rows = length(oid2lid)

    # Allocate PetscMat
    # I don't why I have to set the size like this, but this is the only combination that works
    array = create_matrix(
        comm;
        nrows_loc = n_own_rows,
        ncols_loc = n_own_rows,
        autosetup = true,
    )

    # Set exact preallocation
    _I, _J, _ = findnz(A)
    my_part = MPI.Comm_rank(comm) + 1
    owned_by_me = [part == my_part for part in lid2part]

    # Ugly to allocate this here, will improve later
    coo_I0 = PetscInt[] # 0-based
    coo_J0 = PetscInt[] # 0-based

    # CSR or COO preallocation
    if CSR
        # CSR version
        d_nnz, o_nnz = _preallocation_from_sparse(_I, _J, owned_by_me, oid2lid, lid2pid)
        preallocate!(array, d_nnz, o_nnz)
    else
        # COO version
        sizehint!(coo_I0, length(_I))
        sizehint!(coo_J0, length(_J))
        for (li, lj) in zip(_I, _J)
            owned_by_me[li] || continue
            push!(coo_I0, lid2pid[li] - 1)
            push!(coo_J0, lid2pid[lj] - 1)
        end
        setPreallocationCOO(array, length(coo_I0), coo_I0, coo_J0)
        setOption(array, MAT_NEW_NONZERO_ALLOCATION_ERR, true)
        @show coo_I0
        @show coo_J0
        @show length(coo_I0)
    end

    return array, coo_I0, coo_J0
end

function HauntedArrays.copy_cache(cache::PetscCache)
    if cache.array isa Vec
        array = duplicate(cache.array)
    elseif cache.array isa Mat
        array = duplicate(cache.array, MAT_DO_NOT_COPY_VALUES)
    else
        error("cached array must be of type Vec or Mat")
    end
    return PetscCache(
        array,
        cache.lid2pid,
        cache.oid2pid0,
        cache.CSR,
        cache.coo_mask,
        cache.coo_I0,
        cache.coo_J0,
    )
end

PetscWrap.duplicate(::Nothing) = nothing

"""
    get_updated_petsc_array(x::HauntedVector)
    get_updated_petsc_array(A::HauntedMatrix)

Return a cached Petsc vector/matrix with the values of the input HauntedVector/HauntedMatrix

TODO : for matrix, use cache to avoid reallocation
"""
function get_updated_petsc_array(x::HauntedVector)
    cache = get_cache(x)
    if !(cache isa PetscCache)
        comm = get_comm(x)
        @only_root println(
            "WARNING : no cache found for HauntedVector in `get_updated_petsc_array`",
        ) comm
        cache = HauntedArrays.build_cache(
            PetscCache,
            parent(x),
            HauntedArrays.get_exchanger(x),
            local_to_global(x),
            local_to_part(x),
            own_to_local(x),
        )
    end

    y = cache.array
    update!(y, x, cache.oid2pid0)
    return y
end

function get_updated_petsc_array(A::HauntedMatrix)
    cache = get_cache(A)
    if !(cache isa PetscCache)
        comm = get_comm(A)
        @only_root println(
            "WARNING : no cache found for HauntedMatrix in `get_updated_petsc_array`",
        ) comm
        cache = HauntedArrays.build_cache(
            PetscCache,
            parent(A),
            HauntedArrays.get_exchanger(A),
            local_to_global(A),
            local_to_part(A),
            own_to_local(A),
        )
    end

    B = cache.array
    lid2pid = cache.lid2pid
    coo_mask = cache.coo_mask

    # zeroEntries(B) # this should not be necessary since we fill all values
    if cache.CSR
        update_CSR!(B, A, lid2pid)
    else
        update_COO!(B, A, coo_mask)
    end

    return B
end

"""
Return the Petsc vector that is already allocated. Warning : the returned
vector does NOT correspond to `x` (use get_updated_petsc_array for this).
"""
function get_petsc_array(x::HauntedArray)
    cache = get_cache(x)
    if !(cache isa PetscCache)
        comm = get_comm(A)
        @only_root println("WARNING : no cache found for HauntedArray in `get_petsc_array`") comm
        cache = HauntedArrays.build_cache(
            PetscCache,
            parent(x),
            HauntedArrays.get_exchanger(x),
            local_to_global(x),
            local_to_part(x),
            own_to_local(x),
        )
    end

    return cache.array
end

"""
Find number of non-zeros element per diagonal block and off-diagonal block
(see https://petsc.org/release//manualpages/Mat/MatMPIAIJSetPreallocation/)
"""
function _preallocation_from_sparse(I, J, owned_by_me::Vector{Bool}, oid2lid, lid2pid)
    # Allocate
    nrows = length(oid2lid)
    d_nnz = zeros(Int, nrows)
    o_nnz = zeros(Int, nrows)

    iglob_min, iglob_max = extrema(view(lid2pid, oid2lid))

    # Search for non-zeros
    for (li, lj) in zip(I, J)
        # Check that the row is handled by the local processor
        owned_by_me[li] || continue

        # Global Petsc row number
        iglob = lid2pid[li]

        # Look if the column belongs to a diagonal block or not
        # Rq: `iglob - iglob_min + 1` and not `iloc` because the ghost
        # dofs are not necessarily at the end, they can be any where in local
        # numbering...
        jglob = lid2pid[lj]
        if (iglob_min <= jglob <= iglob_max)
            d_nnz[iglob - iglob_min + 1] += 1
        else
            o_nnz[iglob - iglob_min + 1] += 1
        end
    end
    return d_nnz, o_nnz
end
