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
    update!(
        y::Mat,
        x::HauntedArray{T,2,S},
        coo_mask::Vector{I},
    ) where {T,S<:AbstractSparseArray,I}

Update Petsc Vec with values of HauntedArray

`oid2pid0` is the HauntedVector owned indices to Petsc global indices (0-based for Petsc)

Benchmarked as the fastest solution to achieve this.
"""
function update!(y::Vec, x::HauntedVector, oid2pid0::Vector{PetscInt})
    setValues(y, oid2pid0, owned_values(x), INSERT_VALUES)
    assemble!(y)
end

function update!(y::Mat, x::HauntedArray{T,2,S}, lid2pid) where {T,S<:Matrix}
    _x = parent(x)
    ncols_l = size(_x, 2)
    for li in own_to_local_rows(x)
        set_values!(y, lid2pid[li] .* ones(ncols_l), lid2pid, _x[li, :], ADD_VALUES)
    end
    assemble!(y)
end

function update_COO!(
    y::Mat,
    x::HauntedArray{T,2,S},
    coo_mask::Vector{I},
) where {T,S<:AbstractSparseArray,I}
    _, _, _V = findnz(parent(x))
    println("0")
    A2 = create_matrix(get_comm(x), nrows_loc = 3, ncols_loc = 3, autosetup = true)
    setPreallocationCOO(A2, 5, PetscInt[0, 1, 1, 2, 2], PetscInt[0, 0, 1, 1, 2])
    setValuesCOO(A2, _V[coo_mask], INSERT_VALUES)
    # setValuesCOO(A2, PetscScalar[1, 2, 3, 4, 5], INSERT_VALUES)
    println("0.1")
    zeroEntries(y)
    println("0.2")
    assemble!(y)
    println("0.3")
    matView(y)
    println("entry update!")
    display(parent(x))
    display(_V)
    println("1")
    @show coo_mask
    println("1.1")
    @show _V[coo_mask]
    println("1.2")
    if length(coo_mask) == 5
        _a = PetscScalar[1, 2, 3, 4, 5]
        setValuesCOO(y, _a, INSERT_VALUES)
    end
    println("1.3")
    setValuesCOO(y, _V[coo_mask], INSERT_VALUES)
    println("2")
    assemble!(y)
    println("sortie update!")
end

function update_CSR!(
    y::Mat,
    x::HauntedArray{T,2,S},
    lid2pid,
) where {T,S<:AbstractSparseArray}
    # Retrieve CSR information from SparseArray
    _I, _J, _V = findnz(parent(x))

    # Fill matrix
    for (li, lj, v) in zip(_I, _J, _V)
        owned_by_me(x, li) || continue
        set_value!(y, lid2pid[li], lid2pid[lj], v, ADD_VALUES)
    end
    assemble!(y)
end