# see https://docs.sciml.ai/LinearSolve/dev/solvers/solvers/#HYPRE.jl
# for a note on how to use custom LinearSolver and preconditioners
#
# or https://docs.sciml.ai/LinearSolve/dev/advanced/developing/
#
#
# and https://docs.sciml.ai/LinearSolve/dev/advanced/custom/
module LinearTransport
using MPI
using MPIUtils
using HauntedArrays
using OrdinaryDiffEq
using DiffEqBase
using HauntedArrays2PetscWrap
using LinearSolve
using LinearAlgebra
using SciMLBase

const lx = 1.0
const nx = 4 # on each process
const c = 1.0

function DiffEqBase.recursive_length(A::HauntedVector)
    MPI.Allreduce(n_own_rows(A), MPI.SUM, get_comm(A))
end

function my_linsolve!(
    A::OrdinaryDiffEq.WOperator{IIP},
    b,
    u,
    p,
    newA,
    Pl,
    Pr,
    solverdata;
    verbose = true,
    kwargs...,
) where {IIP}
    @show typeof(A.mass_matrix)
    @show typeof(A.gamma)
    @show typeof(A.J)
    # @show typeof(convert(AbstractMatrix, A.J))
    @show typeof(A._concrete_form)
    @show A.transform
    @show IIP
    error("toto")
    A = convert(AbstractMatrix, A)
    u .= LinearAlgebra.ldiv!(u, A, b)
    return u
end

# struct MyLUFactorization <: SciMLBase.AbstractLinearAlgorithm end
struct MyLUFactorization <: LinearSolve.AbstractFactorization end
# LinearSolve.needs_concrete_A(::MyLUFactorization) = true

function LinearSolve.init_cacheval(
    alg::MyLUFactorization,
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
end

# function SciMLBase.solve(cache::LinearCache, alg::MyLUFactorization; kwargs...)
function SciMLBase.solve(cache::LinearSolve.LinearCache, alg::MyLUFactorization; kwargs...)
    # Version for LinearSolve.AbstractFactorization :
    # if cache.isfresh
    #     fact = do_factorization(alg, cache.A, cache.b, cache.u)
    #     cache = set_cacheval(cache, fact)
    # end
    # y = ldiv!(cache.u, cache.cacheval, cache.b)

    y = ldiv!(cache.u, cache.A, cache.b)
    # @show typeof(cache.u)
    # @show typeof(cache.A)
    # @show typeof(cache.b)
    # @show typeof(y)
    # error("debug")
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end


MPI.Initialized() || MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
np = MPI.Comm_size(comm)

mypart = rank + 1
Δx = lx / (np * nx - 1)

# The "mesh" partitioning
lid2gid, lid2part = HauntedArrays.generate_1d_partitioning(nx, mypart, np)

# Allocate
q = HauntedVector(comm, lid2gid, lid2part; cacheType = PetscCache)
dq = similar(q)
p = (c = c, Δx = Δx)

# Init
(mypart == 1) && (q[1] = 1.0)

function f!(dq, q, p, t)
    # @one_at_a_time println("before update")
    update_ghosts!(q)
    for i in own_to_local(q)
        (local_to_global(q, i) == 1) && continue # boundary condition
        dq[i] = -c / p.Δx * (q[i] - q[i - 1])
    end
end

tspan = (0.0, 2.0)
prob = ODEProblem(f!, q, tspan, p)
# alg = LinearSolveFunction(my_linsolve!)
alg = MyLUFactorization()
# sol = solve(prob, ImplicitEuler(linsolve = alg))
sol = solve(prob, ImplicitEuler())
# sol = solve(prob, Euler(); dt = Δx / c)
q = sol.u[end]

@one_at_a_time @show owned_values(q)

isinteractive() || MPI.Finalize()
end