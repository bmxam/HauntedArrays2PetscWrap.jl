module LinearTransport
using MPI
using MPIUtils
using HauntedArrays
using OrdinaryDiffEq
using HauntedArrays2PetscWrap
using LinearAlgebra
using Symbolics
using SparseDiffTools

# Settings
const lx = 1.0
const nx = 4 # on each process
const c = 1.0
const α = 1.0 # Dirichlet boundary condition
const tspan = (0.0, 5.0)

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
p = (c = c, Δx = Δx, α = α)

function f!(dq, q, p, t)
    # Loop on own dofs
    for i in own_to_local(q)
        if local_to_global(q, i) == 1
            dq[i] = p.α - q[i] # Dirichlet boundary condition
        else
            dq[i] = -p.c / p.Δx * (q[i] - q[i - 1])
        end
    end
end

always_true(args...) = true

# This callback avoids using `update_ghosts!` in `f`, which
# can cause troubles when dealing with Dual elements
cb_update = DiscreteCallback(
    always_true,
    integrator -> begin
        update_ghosts!(integrator.u)
    end;
    save_positions = (false, false),
)


function run_expl(q)
    prob = ODEProblem(f!, q, tspan, p)
    timestepper = Euler()
    @only_root println("running solve (explicit)...")
    sol = solve(prob, timestepper; dt = Δx / c, callback = CallbackSet(cb_update))
    q = sol.u[end]
    @one_at_a_time @show owned_values(q)
end

function run_impl_dense(q)
    prob = ODEProblem(f!, q, tspan, p)
    timestepper = ImplicitEuler(linsolve = PetscFactorization())
    @only_root println("running solve (implicit dense)...")
    sol = solve(prob, timestepper; callback = CallbackSet(cb_update))
    q = sol.u[end]
    @one_at_a_time @show owned_values(q)
end

function run_impl_sparse(q)
    input = similar(q)
    output = similar(q)
    _f! = (y, x) -> f!(y, x, p, 0.0)
    sparsity_pattern = Symbolics.jacobian_sparsity(_f!, output, input)
    jac = HauntedMatrix(Float64.(sparsity_pattern), q)
    colors = matrix_colors(jac)

    ode = ODEFunction(f!; jac_prototype = jac, colorvec = colors)
    prob = ODEProblem(ode, q, tspan, p)
    timestepper = ImplicitEuler(linsolve = PetscFactorization())
    @only_root println("running solve (implicit sparse)...")
    sol = solve(prob, timestepper; callback = CallbackSet(cb_update))
    q = sol.u[end]
    @one_at_a_time @show owned_values(q)
end

run_expl(q)
run_impl_dense(q)
run_impl_sparse(q)

isinteractive() || MPI.Finalize()
end
