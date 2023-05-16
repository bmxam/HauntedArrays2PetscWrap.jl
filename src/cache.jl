const DEFAULT_IS_CSR = false

"""
WIP on the structure : for now I store everything I need for every solution,
will remove a lot of stuff later
"""
struct PetscCache{P,I} <: HauntedArrays.AbstractCache
    # Petsc.Vec, Petsc.Mat or Nothing
    array::P

    # Use CSR or COO? -> if we keep this, use Trait or type param
    is_CSR::Bool

    # Common infos
    lid2pid0::Vector{PetscInt} # Local to PETSc indexing (0-based)
    oid2pid0::Vector{PetscInt} # own to PETSc indexing (0-based)

    # CSR infos
    rowptr::Vector{PetscInt} # 1-based
    colval # (Vector or SubArray) 1-based
    perm::Vector{I} # 1-based, permutation of `V` to use with CSR

    # COO infos
    coo_mask::Vector{I} # Index of owned values in I, J, V (only for sparse matrices)
    coo_I0::Vector{PetscInt} # 0-based
    coo_J0::Vector{PetscInt} # 0-based


    function PetscCache(
        a,
        is_CSR::Bool,
        lid2pid0::Vector{PetscInt},
        oid2pid0::Vector{PetscInt},
        rowptr::Vector{PetscInt},
        colval::AbstractVector{I},
        perm::Vector{I},
        coo_mask::Vector{I},
        coo_I0::Vector{PetscInt},
        coo_J0::Vector{PetscInt},
    ) where {I<:Integer}
        new{typeof(a),I}(
            a,
            is_CSR,
            lid2pid0,
            oid2pid0,
            rowptr,
            colval,
            perm,
            coo_mask,
            coo_I0,
            coo_J0,
        )
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
    is_CSR = haskey(kwargs, :is_CSR) ? kwargs[:is_CSR] : DEFAULT_IS_CSR
    # @assert is_CSR "COO not working for now"

    # Number of elements handled by each proc
    n_by_rank = MPI.Allgather(length(oid2lid), comm)
    # nrows_glob = sum(n_by_rank)

    # Init Petsc if needed (should use options here)
    if !PetscInitialized()
        @warn "PETSc has not been initialized, you should call `init_petsc` before creating a cache"
        PetscInitialize()
    end

    # Compute local to petsc numbering
    lid2pid = _compute_lid2pid(exchanger, n_by_rank, lid2gid, oid2lid)
    lid2pid0 = PetscInt.(lid2pid .- 1)
    oid2pid0 = lid2pid0[oid2lid] # allocation is wanted

    # Compute csr infos
    if array isa AbstractSparseMatrix
        _rowptr, colval, perm = csc_to_csr(array)
        rowptr = PetscInt.(_rowptr)
    else
        rowptr = PetscInt[]
        colval = Int[]
        perm = Int[]
    end

    # Compute coo infos
    coo_mask = Int[]
    if array isa AbstractSparseMatrix
        _I, _, _ = findnz(array)
        coo_mask = findall(li -> lid2part[li] == my_part, _I)
        # For now `coo_mask` is column-wise ordered (because coming from CSC),
        # we need to order it row-wise for petsc
        coo_mask = coo_mask[sortperm(_I[coo_mask])]
        # coo_mask .= view(coo_mask, sortperm(view(_I, coo_mask))) # should work, but need to be validated
    end

    # Build Petsc Array
    _array, coo_I0, coo_J0 =
        _build_petsc_array(comm, array, lid2part, oid2lid, lid2pid, is_CSR)

    return PetscCache(
        _array,
        is_CSR,
        lid2pid0,
        oid2pid0,
        rowptr,
        colval,
        perm,
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
    lid2pid = zero(lid2gid) # `zero` ensures (later) that all elements are set
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
    vec = create_vector(comm; nrows_loc = n_own_rows, autosetup = true)
    coo_I0 = PetscInt[]
    coo_J0 = PetscInt[]
    return vec, coo_I0, coo_J0
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
    mat = create_matrix(
        comm;
        nrows_loc = n_own_rows,
        ncols_loc = n_own_rows,
        autosetup = true,
    )
    coo_I0 = PetscInt[]
    coo_J0 = PetscInt[]
    return mat, coo_I0, coo_J0
end

function _build_petsc_array(
    comm::MPI.Comm,
    A::AbstractSparseArray{T},
    lid2part,
    oid2lid,
    lid2pid,
    is_CSR::Bool,
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
    if is_CSR
        # CSR version
        d_nnz, o_nnz = _preallocation_CSR(_I, _J, owned_by_me, oid2lid, lid2pid)
        preallocate!(array, d_nnz, o_nnz)
    else
        coo_I0, coo_J0 = _preallocation_COO(_I, _J, owned_by_me, lid2pid)
        setPreallocationCOO(array, length(coo_I0), coo_I0, coo_J0)
        setOption(array, MAT_NEW_NONZERO_ALLOCATION_ERR, true)
    end

    return array, coo_I0, coo_J0
end

function HauntedArrays.copy_cache(cache::PetscCache)
    if cache.array isa Vec
        array = duplicate(cache.array)
    elseif cache.array isa Mat
        array = duplicate(cache.array, MAT_DO_NOT_COPY_VALUES)

        # Due to a bug in PETSc, it's necessary to call again setPreallocationCOO
        if cache.is_CSR == false
            setPreallocationCOO(array, cache.coo_I0, cache.coo_J0)
        end
    else
        error("cached array must be of type Vec or Mat")
    end
    return PetscCache(
        array,
        cache.is_CSR,
        cache.lid2pid0,
        cache.oid2pid0,
        cache.rowptr,
        cache.colval,
        cache.perm,
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

    # zeroEntries(B) # this should not be necessary since we fill all values
    if cache.is_CSR
        update_CSR!(B, A, cache.lid2pid0, cache.rowptr, cache.colval, cache.perm)
    else
        update_COO!(B, A, cache.coo_mask)
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

