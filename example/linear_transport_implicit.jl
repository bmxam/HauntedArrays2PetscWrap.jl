module LinearTransport
using MPI
using MPIUtils
using HauntedArrays
using OrdinaryDiffEq
using HauntedArrays2PetscWrap
using LinearAlgebra

# Settings
const lx = 1.0
const nx = 4 # on each process
const c = 1.0

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
    for i in own_to_local(q)
        (local_to_global(q, i) == 1) && continue # boundary condition
        dq[i] = -c / p.Δx * (q[i] - q[i - 1])
    end
end

tspan = (0.0, 2.0)
prob = ODEProblem(f!, q, tspan, p)

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

@only_root println("running solve...")
timestepper = ImplicitEuler(linsolve = PetscFactorization())
sol = solve(prob, timestepper; callback = CallbackSet(cb_update))
q = sol.u[end]

@one_at_a_time @show owned_values(q)

isinteractive() || MPI.Finalize()
end