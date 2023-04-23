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
    # do nothing for now
    # TODO : store the PetscMat in the cache
end

function SciMLBase.solve(cache::LinearSolve.LinearCache, alg::PetscFactorization; kwargs...)
    # Version for LinearSolve.AbstractFactorization :
    # if cache.isfresh
    #     fact = do_factorization(alg, cache.A, cache.b, cache.u)
    #     cache = set_cacheval(cache, fact)
    # end
    # y = ldiv!(cache.u, cache.cacheval, cache.b)

    y = ldiv!(cache.u, cache.A, cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end
