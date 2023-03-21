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
using LinearAlgebra

const lx = 1.0
const nx = 3 # on each process
const c = 1.0

function DiffEqBase.recursive_length(A::HauntedVector)
    MPI.Allreduce(n_own_rows(A), MPI.SUM, get_comm(A))
end

#### DEBUG BELOW
function LinearAlgebra.lu(A::HauntedMatrix{T}, pivot; check) where {T}
    return LinearAlgebra.lu(parent(A), pivot, check) # this is wrong, only for debugging
end
#### DEBUG BELOW

MPI.Initialized() || MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
np = MPI.Comm_size(comm)

mypart = rank + 1
Δx = lx / (np * nx - 1)

lid2gid = collect((rank * nx + 1):((rank + 1) * nx))
lid2part = mypart .* ones(Int, nx)
if mypart != np
    append!(lid2gid, (rank + 1) * nx + 1)
    append!(lid2part, mypart + 1)
end
if mypart != 1
    prepend!(lid2gid, rank * nx)
    prepend!(lid2part, mypart - 1)
end

@one_at_a_time @show lid2gid
@one_at_a_time @show lid2part

# Allocate
q = HauntedVector(comm, lid2gid, lid2part)
dq = similar(q)
p = (c = c, Δx = Δx)

# Init
(mypart == 1) && (q[1] = 1.0)

function f!(dq, q, p, t)

    update_ghosts!(q)
    for i in own_to_local(q)
        (local_to_global(q, i) == 1) && continue # boundary condition
        dq[i] = -c / p.Δx * (q[i] - q[i - 1])
    end
end

tspan = (0.0, 2.0)
prob = ODEProblem(f!, q, tspan, p)

# Implicit time integration
sol = solve(prob, ImplicitEuler())
q = sol.u[end]

update_ghosts!(q)
@one_at_a_time @show owned_values(q)

isinteractive() || MPI.Finalize()
end