struct PetscFactorization <: LinearSolve.AbstractFactorization
    options

    PetscFactorization(options = Dict()) = new(options)
end

function LinearSolve.init_cacheval(
    alg::PetscFactorization,
    A,
    b,
    u,
    Pl,
    Pr,
    maxiters::Int,
    abstol,
    reltol,
    verbose::Bool,
    assumptions::LinearSolve.OperatorAssumptions,
)
    # here A and b are random (even null), this is just to allocate the cache. So we just need to create a ksp without
    # calling `setup` otherwise the PETSc factorization is triggered (leading to a zero pivot)
    # Note : we could even just return an empty KSP...
    _A = get_updated_petsc_array(A)
    return create_ksp(_A; autosetup = false)
end

function LinearSolve.do_factorization(::PetscFactorization, A, b, u)
    _A = get_updated_petsc_array(A)
    return create_ksp(_A; autosetup = true)
end

# function SciMLBase.solve(cache::LinearSolve.LinearCache, alg::PetscFactorization; kwargs...)
#     # Version for LinearSolve.AbstractFactorization :
#     # if cache.isfresh
#     #     fact = do_factorization(alg, cache.A, cache.b, cache.u)
#     #     cache = set_cacheval(cache, fact)
#     # end
#     # y = ldiv!(cache.u, cache.cacheval, cache.b)

#     y = ldiv!(cache.u, cache.A, cache.b)
#     SciMLBase.build_linear_solution(alg, y, nothing, cache)
# end
