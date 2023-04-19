# This file is under (heavy) construction

function LinearAlgebra.ldiv!(x::HauntedVector, A::HauntedMatrix, b::HauntedVector)
    # Convert to PETSc objects
    _A = get_updated_petsc_array(A)
    _b = get_updated_petsc_array(b)
    # _x = get_cached_vector(x)

    ksp = create_ksp(_A; autosetup = true)

    # Solve the system
    _x = PetscWrap.solve(ksp, _b)
    # PetscWrap.solve(ksp, _b, _x)

    # Convert `Vec` to Julia `Array` (memory leak here?)
    # TODO : implement the version with a loop and VecGetValues
    x[own_to_local_rows(x)] .= vec2array(_x)

    # Free memory (to be improved)
    # release_cache(get_cache(x); free = true)
    # destroy!(_x)

    if get_cache(A) isa PetscCache
        zeroEntries(_A)
    else
        destroy!(_A)
    end

    if get_cache(b) isa PetscCache
        # do nothing
    else
        destroy!(_b)
    end

    return x
end
