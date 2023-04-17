# This file is under (heavy) construction

function LinearAlgebra.ldiv!(x::HauntedVector, A::HauntedMatrix, b::HauntedVector)
    # Convert to PETSc objects
    _A = get_updated_petsc_array(A)
    _b = get_updated_petsc_array(b)

    ksp = create_ksp(_A; autosetup=true)

    # Solve the system
    _x = PetscWrap.solve(ksp, _b)

    # Convert `Vec` to Julia `Array` (memory leak here?)
    x[own_to_local_rows(x)] .= vec2array(_x)

    # Free memory (_A and _b may be cached and should not be destroyed here)
    destroy!(_x)

    return x
end