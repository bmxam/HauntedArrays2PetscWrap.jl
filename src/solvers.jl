"""
In the future, if needed : replace all attributes by a Dict
"""
struct PetscFactorization <: LinearSolve.AbstractFactorization
    ksp_finalizer::Bool
end

PetscFactorization(; ksp_finalizer = false) = PetscFactorization(ksp_finalizer)

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
    # here A and b are random (even null), this is just to allocate the cache. So we just
    # need to create a ksp without calling `setup` otherwise the PETSc factorization is
    # triggered (leading to a zero pivot)
    # Note : we could even just return an empty KSP...
    _A = get_updated_petsc_array(A)
    return create_ksp(_A; autosetup = false, add_finalizer = alg.ksp_finalizer)
end

function LinearSolve.do_factorization(alg::PetscFactorization, A, b, u)
    _A = get_updated_petsc_array(A)
    return create_ksp(_A; autosetup = true, add_finalizer = alg.ksp_finalizer)
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
