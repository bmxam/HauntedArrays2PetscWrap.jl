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
Update Petsc Vec with values of HauntedVector

`oid2pid0` is the HauntedVector owned indices to Petsc global indices (0-based for Petsc)

Benchmarked as the fastest solution to achieve this.
"""
function update!(y::Vec, x::HauntedVector, oid2pid0::Vector{PetscInt})
    setValues(y, oid2pid0, owned_values(x), INSERT_VALUES)
    assemble!(y)
end