# This file is under (heavy) construction
#-> I will introduce a PetscFactorization <: LinearAlgebra.Factorization
# later

function LinearAlgebra.lu!(A::HauntedMatrix, pivot; check = true)
    println("wrong `lu!` for debug")
    # don't do anything for now
    return A
end

function LinearAlgebra.generic_lufact!(A::HauntedMatrix, pivot; check = true)
    println("wrong `generic_lufact!` for debug")
    # don't do anything for now
    return A
end

function LinearAlgebra.:\(A::HauntedMatrix, b::HauntedVector)
    x = similar(b)
    ldiv!(x, A, b)
    return x
end

function LinearAlgebra.ldiv!(
    x::HauntedVector,
    A::LinearAlgebra.Factorization,
    b::HauntedVector,
)
    println("wrong `LinearAlgebra.ldiv!` for debug (should pass here if using PetscSolver)")
    # function LinearAlgebra.ldiv!(x::HauntedVector, A::LinearAlgebra.LU, b::HauntedVector)
    # THIS IS A TEMPORARY HACK TO "IGNORE" FACTORIZATION
    ldiv!(x, A.factors, b)
    return x
end

function LinearAlgebra.ldiv!(x::HauntedVector, A::HauntedMatrix, b::HauntedVector)
    # Convert to PETSc objects
    _A = get_updated_petsc_array(A)
    _b = get_updated_petsc_array(b)

    ksp = create_ksp(_A; autosetup = true)

    # Solve the system
    _x = solve(ksp, _b)

    # Convert `Vec` to Julia `Array` (memory leak here?)
    x[own_to_local_rows(x)] .= vec2array(_x)

    # Free memory (_A and _b may be cached and should not be destroyed here)
    destroy!(_x)

    return x
end