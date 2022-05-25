module ChebUtils

using LinearAlgebra

export chebpts, chebdiff, chebddiff, chebws

"""
    Generate a set of Chebyshev points.
"""
chebpts(Ny::Int, ::Type{T}=Float64) where {T} = T.(cos.((0:(Ny - 1))π/(Ny - 1)))

"""
    Calculate the first order Chebyshev differentiation matrix for a given
    number of Chebyshev discretisation points.

    R. Baltensperger, J. P. Berrut, The error in calculating the pseudospectral
    differentaition matrices for Chebyshev-Gauss-Lobatto points.
"""
function chebdiff(Ny::Int, ::Type{T}=Float64) where {T}
    # initialise matrix
    diffmat = zeros(T, Ny, Ny)

    # define anonymous function for c_k coefficient and x location
    c = i -> i == 1 || i == Ny ? 2 : 1
    x = i -> cos(((i - 1)*π)/(Ny - 1))

    # loop over matrix assigning values
    for i in 1:Ny, j in 1:Ny
        if i != j
            diffmat[i, j] = (c(i)/c(j))*(((-1)^(i + j))/(x(i) - x(j)))
        end
    end

    # compute diagonal entries
    for i in 1:Ny
        diffmat[i, i] = -sum([diffmat[i, j] for j in 1:Ny if i != j])
    end

    return diffmat
end

"""
    Calculate the second order Chebyshev differentiation matrix for a given
    number of Chebyshev discretisation points.

    R. Baltensperger, J. P. Berrut, The error in calculating the pseudospectral
    differentaition matrices for Chebyshev-Gauss-Lobatto points.
"""
function chebddiff(Ny::Int, ::Type{T}=Float64) where {T}
    # construct first derivative matrix
    Dy = chebdiff(Ny, T)

    # call other method for second derivative matrix
    chebddiff(Dy)
end

function chebddiff(Dy::Matrix{T}) where {T}
    # initialise matrix
    diffmat = zero(Dy)

    # define anonymous function for c_k coefficients and x location
    c = i -> i == 1 || i == size(Dy)[1] ? 2 : 1
    x = i -> cos(((i - 1)*π)/(size(Dy)[1] - 1))

    # loop over matrix assigning values
    for i in 1:size(Dy)[1], j in 1:size(Dy)[1]
        if i != j
            diffmat[i, j] = 2*Dy[i, j]*(Dy[i, i] - 1/(x(i) - x(j)))
        else
            diffmat[i, i] = 2*(Dy[i, i]^2 + sum([Dy[i, k]/(x(i) - x(k)) for k in 1:size(Dy)[1] if k != i]))
        end
    end

    return diffmat
end

"""
    Calculate the quadrature weights for a given Chebyshev grid size, based on
    the Clenshaw-Curtis quadrature method.
"""
chebws(Dy::Matrix{T}) where {T} = append!(inv(Dy[1:(end - 1), 1:(end - 1)])[1, :], [0])

end
