# This file is under (heavy) construction

function LinearAlgebra.lu!(A::HauntedMatrix, pivot; check)
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

function LinearAlgebra.ldiv!(x::HauntedVector, A::HauntedMatrix, b::HauntedVector)
    # DEBUG version !
    # @assert MPI.Comm_size(get_comm(A)) == 1 "invalid ldiv! on nprocs > 1"
    # x .= parent(A) \ parent(b)
    # return x

    # Convert to PETSc objects
    _A = get_updated_petsc_array(A)
    _b = get_updated_petsc_array(b)

    ksp = create_ksp(_A; autosetup = true)

    # Solve the system
    _x = solve(ksp, _b)

    # Convert `Vec` to Julia `Array` (memory leak here?)
    x .= vec2array(_x)

    # Free memory (_A and _b may be cached and should not be destroyed here)
    destroy!(_x)

    return x
end