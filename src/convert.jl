function get_updated_petsc_array(x::HauntedVector)
    # Use cache to retrieve PetscVec and lid2pid, or recompute it
    cache = get_cache(x)
    if cache isa PetscCache
        lid2pid = cache.lid2pid
        y = cache.array
    else
        @only_root println(
            "WARNING : no cache found for HauntedVector in `get_updated_petsc_array`",
        )
        n_by_rank = MPI.Allgather(length(x.oid2lid), get_comm(x))

        lid2pid = _compute_lid2pid(
            HauntedArrays.get_exchanger(x),
            n_by_rank,
            local_to_global(x),
            own_to_local(x),
        )
        y = _build_petsc_array(get_comm(x), n_own_rows(x), ndims(x))
    end
    set_values!(y, lid2pid[own_to_local(x)], owned_values(x))
    assemble!(y)
    return y
end

"""
WIP : use cache to avoid reallocation
"""
function get_updated_petsc_array(A::HauntedMatrix)
    # Alias
    _A = parent(A)

    (_A isa Array) || error("HauntedArray with parent # Array not implemented yet")

    # Use cache to lid2pid, or recompute it
    cache = get_cache(A)
    if cache isa PetscCache
        lid2pid = cache.lid2pid
        ncols_g = cache.nrows_glob
    else
        @only_root println(
            "WARNING : no cache found for HauntedMatrix in `get_updated_petsc_array`",
        )

        n_by_rank = MPI.Allgather(length(A.oid2lid), get_comm(A))
        ncols_g = sum(n_by_rank)

        lid2pid = _compute_lid2pid(
            HauntedArrays.get_exchanger(A),
            n_by_rank,
            local_to_global(A),
            own_to_local(A),
        )
    end

    # Allocate PetscMat and fill it
    B = create_matrix(
        get_comm(A);
        nrows_loc = n_own_rows(A),
        ncols_glo = ncols_g,
        autosetup = true,
    )
    _fill_petscmat_with_array!(B, A, lid2pid)

    assemble!(B)

    return B
end

function _fill_petscmat_with_array!(B::Mat, A::HauntedArray{T,2,Array}, lid2pid) where {T}
    _A = parent(A)
    ncols_l = size(_A, 2)
    for li in own_to_local_rows(A)
        set_values!(B, lid2pid[li] .* ones(ncols_l), lid2pid, _A[li, :])
    end
end

"""
TODO:  try to use `set_values!` instead of `set_value!`
"""
function _fill_petscmat_with_array!(B::Mat, A::HauntedArray{T,2,AbstractSparseArray}, lid2pid) where {T}
    _A = parent(A)

    # Retrieve CSR information from SparseArray
    _I, _J, _V = findnz(_A)

    # Set exact preallocation
    d_nnz, o_nnz = _preallocation_from_sparse(I, J, A, lid2pid)
    preallocate!(B, d_nnz, o_nnz)

    # Fill matrix
    for (li, lj, v) in zip(_I, _J, _V)
        owned_by_me(A, li) || continue
        set_value!(B, lid2pid[li], lid2pid[lj], v, mode = ADD_VALUES)
    end
end

"""
Find number of non-zeros element per diagonal block and off-diagonal block
(see https://petsc.org/release/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html)
"""
function _preallocation_from_sparse(
    I,
    J,
    A::HauntedArray,
    lid2pid
)
    # Allocate
    nrows = n_own_rows(A)
    d_nnz = zeros(Int, nrows)
    o_nnz = zeros(Int, nrows)

    iglob_min, iglob_max = extrema(view(lid2pid, own_to_local(A)))

    # Search for non-zeros
    for (li, lj) in zip(I, J)
        # Check that the row is handled by the local processor
        owned_by_me(A, li) || continue

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