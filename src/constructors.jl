# This file contains the methods that form the exposed interface to construct
# Chebyshev differentiation matrices.

function chebdiff(N::Int, ::Type{T}=Float64) where {T}
    # initialise matrix
    D = zeros(T, N, N)

    # define anonymous function for c_k coefficient and x location
    c = i -> i == 1 || i == N ? 2 : 1
    x = i -> cos(((i - 1)*π)/(N - 1))

    # loop over matrix assigning values
    for i in 1:N, j in 1:N
        if i != j
            D[i, j] = (c(i)/c(j))*(((-1)^(i + j))/(x(i) - x(j)))
        end
    end

    # compute diagonal entries
    for i in 1:N
        D[i, i] = -sum([D[i, j] for j in 1:N if i != j])
    end

    return ChebDiff(D, T)
end

function chebddiff(D::ChebDiff{T}) where {T}
    # initialise matrix
    DD = zero(D)

    # define anonymous function for c_k coefficients and x location
    x = i -> cos(((i - 1)*π)/(size(D)[1] - 1))

    # loop over matrix assigning values
    for i in 1:size(D)[1], j in 1:size(D)[1]
        if i != j
            DD[i, j] = 2*D[i, j]*(D[i, i] - 1/(x(i) - x(j)))
        else
            DD[i, i] = 2*(D[i, i]^2 + sum([D[i, k]/(x(i) - x(k)) for k in 1:size(D)[1] if k != i]))
        end
    end

    return DD
end
chebddiff(N::Int, ::Type{T}=Float64) where {T} = chebddiff(chebdiff(N, T))

