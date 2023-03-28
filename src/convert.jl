function get_updated_petsc_array(x::HauntedVector)
    # Use cache to retrieve PetscVec and lid2pid, or recompute it
    cache = get_cache(x)
    if typeof(cache) == PetscCache
        lid2pid = cache.lid2pid
        y = cache.array
    else
        lid2pid = _compute_lid2pid(
            HauntedArrays.get_exchanger(x),
            local_to_global(x),
            own_to_local(x),
        )
        y = _build_petsc_array(get_comm(x), n_own_rows(x), ndims(x))
    end
    set_values!(y, lid2pid[own_to_local(x)], owned_values(x))
    assemble!(y)
    return y
end

function get_updated_petsc_array(A::HauntedMatrix)
    # Alias
    _A = parent(A)

    (_A isa Array) || error("HauntedArray with parent # Array not implemented yet")

    # Use cache to lid2pid, or recompute it
    cache = get_cache(A)
    if typeof(cache) == PetscCache
        lid2pid = cache.lid2pid
    else
        lid2pid = _compute_lid2pid(
            HauntedArrays.get_exchanger(A),
            local_to_global(A),
            own_to_local(A),
        )
    end

    # Allocate PetscMat and fill it
    ncols_l = size(_A, 2)
    B = create_matrix(get_comm(A), n_own_rows(A), ncols_l)
    setFromOptions(B)
    setUp(B)
    for li in own_to_local_rows(A)
        set_values!(B, lid2pid[li], lid2pid, _A[li, :])
    end
    assemble!(B)

    return B
end