# This file is under (heavy) construction

function LinearAlgebra.:(\)(A::HauntedMatrix, b::HauntedVector)
    x = similar(b)
    return ldiv!(x, A, b)
end

function LinearAlgebra.ldiv!(x::HauntedVector, A::HauntedMatrix, b::HauntedVector)
    # Convert to PETSc objects
    _A = get_updated_petsc_array(A)
    _b = get_updated_petsc_array(b)
    _x = get_petsc_array(x)

    ksp = create_ksp(_A; autosetup = true)

    # Solve the system
    # _x = PetscWrap.solve(ksp, _b)
    PetscWrap.solve(ksp, _b, _x)

    # Update HauntedVector `x` with values of Petsc Vec `_x`
    update!(x, _x)

    if !(get_cache(A) isa PetscCache)
        destroy!(_A)
    end

    if !(get_cache(b) isa PetscCache)
        destroy!(_b)
    end

    return x
end

function LinearAlgebra.ldiv!(x::HauntedVector, ksp::KSP, b::HauntedVector)
    # Convert to PETSc objects
    _b = get_updated_petsc_array(b)
    _x = get_petsc_array(x)

    # Solve the system
    # _x = PetscWrap.solve(ksp, _b)
    PetscWrap.solve(ksp, _b, _x) # TODO : why is this solution not used?

    # Update HauntedVector `x` with values of Petsc Vec `_x`
    update!(x, _x)

    if !(get_cache(b) isa PetscCache)
        destroy!(_b)
    end

    return x
end
