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
using LinearSolve
using LinearAlgebra
using PetscWrap
using HauntedArrays2PetscWrap

const lx = 1.0
const nx = 3 # on each process
const c = 1.0

function DiffEqBase.recursive_length(A::HauntedVector)
    MPI.Allreduce(n_own_rows(A), MPI.SUM, get_comm(A))
end

#### DEBUG BELOW
function LinearAlgebra.lu!(A::HauntedMatrix{T}, pivot; check) where {T}
    println("wrong `lu!` for debug")
    # don't do anything for now
    return A
end

function LinearAlgebra.generic_lufact!(A::HauntedMatrix{T}, pivot; check = true) where {T}
    println("wrong `generic_lufact!` for debug")
    # don't do anything for now
    return A
end

function LinearAlgebra.ldiv!(x::HauntedVector, A::HauntedMatrix, b::HauntedVector)
    # DEBUG version !
    # @assert MPI.Comm_size(get_comm(A)) == 1 "invalid ldiv! on nprocs > 1"
    # x .= parent(A) \ parent(b)
    # return x

    # Convert to PETSc objects
    _A = get_updated_petsc_array(A)
    _b = get_updated_petsc_array(b)

    ksp = create_ksp(_A; auto_setup = true)

    # Solve the system
    _x = solve(ksp, _b)

    # Convert `Vec` to Julia `Array` (memory leak here?)
    x .= vec2array(_x)

    # Free memory (_A and _b may be cached and should not be destroyed here)
    destroy!.(_x)
end

# function my_linsolve(A, b, u, p, newA, Pl, Pr, solverdata; verbose = true, kwargs...)
#     @show typeof(A)
#     @show typeof(b)
#     @show fieldnames(typeof(A))
#     @show A.mass_matrix
#     @show A.gamma
#     @show A.J
#     @show fieldnames(typeof(A.J))
#     @show A.J.cache1
#     @show A.J.cache2
#     @show A.J.autodiff
#     A = convert(AbstractMatrix, A)
#     @show typeof(A)

#     _u = parent(A) \ parent(b)
#     u .= _u
#     return u
# end
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
# alg = LinearSolveFunction(my_linsolve)
# sol = solve(prob, ImplicitEuler(linsolve = alg))
# sol = solve(prob, ImplicitEuler(precs = DEFAULT_PRECS))
sol = solve(prob, ImplicitEuler())
q = sol.u[end]

update_ghosts!(q)
@one_at_a_time @show owned_values(q)

isinteractive() || MPI.Finalize()
end