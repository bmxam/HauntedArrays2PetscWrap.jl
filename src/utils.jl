"""
Update HauntedVector with values of Petsc Vec

Benchmarked as the fastest solution to achieve this.
"""
function update!(y::HauntedVector, x::Vec)
    array_ref = Ref{Ptr{PetscScalar}}()
    error = ccall(
        (:VecGetArray, PetscWrap.libpetsc),
        PetscErrorCode,
        (CVec, Ref{Ptr{PetscScalar}}),
        x,
        array_ref,
    )
    @assert iszero(error)

    arrayFromC = unsafe_wrap(Array, array_ref[], n_own_rows(y); own = false)
    for (i, li) in enumerate(own_to_local(y))
        y[li] = arrayFromC[i]
    end
    restoreArray(x, array_ref)
end

"""
    update!(y::Vec, x::HauntedVector, oid2pid0::Vector{PetscInt})
    update!(y::Mat, x::HauntedArray{T,2,S}, lid2pid) where {T,S<:Matrix}

Update Petsc Vec with values of HauntedArray

`oid2pid0` is the HauntedVector owned indices to Petsc global indices (0-based for Petsc)

Benchmarked as the fastest solution to achieve this.
"""
function update!(y::Vec, x::HauntedVector, oid2pid0::Vector{PetscInt})
    setValues(y, oid2pid0, owned_values(x), INSERT_VALUES)
    assemble!(y)
end

function update!(y::Mat, x::HauntedArray{T,2,S}, lid2pid0) where {T,S<:Matrix}
    # WARNING : not optimized
    _x = parent(x)
    ncols_l = size(_x, 2)
    _ones = ones(ncols_l)
    for li in own_to_local_rows(x)
        setValues(y, lid2pid0[li] .* _ones, lid2pid0, _x[li, :], INSERT_VALUES)
    end
    assemble!(y)
end

function update_COO!(
    y::Mat,
    x::HauntedArray{T,2,S},
    coo_mask::Vector{I},
) where {T,S<:AbstractSparseArray,I}
    _V = nonzeros(parent(x))
    setValuesCOO(y, _V[coo_mask], INSERT_VALUES)
    assemble!(y)
end

"""
it seems there is a bug in this one...
"""
function update_CSR!(
    y::Mat,
    x::HauntedArray{T,2,S},
    lid2pid0::Vector{PetscInt},
    rowptr::Vector{PetscInt},
    colval,
    perm,
) where {T,S<:AbstractSparseArray}
    # Retrieve CSR information from SparseArray
    _V = nonzeros(parent(x))

    petscRow = PetscInt[0]

    # Fill matrix
    for irow in own_to_local(x)
        nelts = rowptr[irow + 1] - rowptr[irow]
        (nelts > 0) || continue
        cols = rowptr[irow]:(rowptr[irow + 1] - 1)
        petscRow[1] = lid2pid0[irow]
        setValues(
            y,
            PetscIntOne,
            petscRow,
            nelts,
            lid2pid0[view(colval, cols)],
            _V[view(perm, cols)],
            INSERT_VALUES,
        )
    end

    assemble!(y)
end

function v1_update_CSR!(
    y::Mat,
    x::HauntedArray{T,2,S},
    lid2pid0,
) where {T,S<:AbstractSparseArray}
    # Retrieve CSR information from SparseArray
    _I, _J, _V = findnz(parent(x))

    row = PetscInt[0]
    col = PetscInt[0]
    val = PetscScalar[0.0]

    # Fill matrix
    for (li, lj, v) in zip(_I, _J, _V)
        owned_by_me(x, li) || continue
        row[1] = lid2pid0[li]
        col[1] = lid2pid0[lj]
        val[1] = v
        setValues(y, PetscIntOne, row, PetscIntOne, col, val, INSERT_VALUES)
    end
    assemble!(y)
end

"""
    csc_to_csr(A::AbstractSparseMatrix)
    csc_to_csr(I::Vector, J::Vector, nrows::Int)

Convert CSC infos to CSR infos
"""
function csc_to_csr(A::AbstractSparseMatrix)
    I, J, _ = findnz(A)
    return csc_to_csr(I, J, size(A, 1))
end

function csc_to_csr(I::Vector, J::Vector, nrows::Int)
    S = sortperm(I)
    perm = copy(S)
    _I = view(I, S)
    _J = view(J, S)
    rowptr = zeros(Int, nrows + 1)
    colval = _J
    i_coo = 1
    rowptr[1] = 1
    for irow = 1:nrows
        nelts = 0
        while (i_coo <= length(_I)) && (_I[i_coo] == irow)
            nelts += 1
            i_coo += 1
        end
        rowptr[irow + 1] = rowptr[irow] + nelts
        if nelts > 0
            r = (i_coo - nelts):(i_coo - 1)
            arg = sortperm(view(colval, r))
            colval[r] .= colval[r[arg]]
            perm[r] .= perm[r[arg]]
        end
    end
    return rowptr, colval, perm
end

"""
Find number of non-zeros element per diagonal block and off-diagonal block
(see https://petsc.org/release//manualpages/Mat/MatMPIAIJSetPreallocation/)
"""
function _preallocation_CSR(I, J, owned_by_me::Vector{Bool}, oid2lid, lid2pid)
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

function _preallocation_COO(I, J, owned_by_me::Vector{Bool}, lid2pid)
    coo_I0 = PetscInt[]
    coo_J0 = PetscInt[]
    sizehint!(coo_I0, length(I))
    sizehint!(coo_J0, length(J))
    for (li, lj) in zip(I, J)
        owned_by_me[li] || continue
        push!(coo_I0, lid2pid[li] - 1)
        push!(coo_J0, lid2pid[lj] - 1)
    end
    return coo_I0, coo_J0
end