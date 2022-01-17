module ChebUtils

import LinearAlgebra

export chebpts, chebdiff, chebddiff, chebws

"""
    Generate a set of Chebyshev points.
"""
chebpts(Ny::Int, ::Type{T}=Float64) where {T} = T.(cos.((0:(Ny - 1))π/(Ny - 1)))

"""
    Calculate the second order Chebyshev differentiation matrix for a given
    number of Chebyshev discretisation points.

    U. Ehrenstein, R. Peyret (1989), A Chebyshev collocation method for the
    Navier-Stokes equations with application to double-diffusive convection.
"""
function chebddiff(Ny::Int, ::Type{T}=Float64) where {T}
    # initialise matrix
    diffmat = Matrix{T}(undef, Ny, Ny)

    # define anonymous function for c_k coefficient and x location
    c = i -> i == 1 || i == Ny ? 2 : 1
    x = i -> cos(((i - 1)*π)/(Ny - 1))

    # loop over matrix assigning values
    for i in 1:Ny, j in 1:Ny
        if i == j
            # evaluate diagonals
            diffmat[i, i] = -((((Ny - 1)^2 - 1)*(1 - x(i)^2) + 3)/(3*(1 - x(i)^2)^2))
        elseif i == 1
            # evaluate top edge (exclusive of diagonal corner)
            diffmat[1, j] = (2/3)*(((-1)^(j - 1))/c(j))*(((2*(Ny - 1)^2 + 1)*(1 - x(j)) - 6)/((1 - x(j))^2))
        elseif i == Ny
            # evaluate bottom edge (exclusive of diagonal corner)
            diffmat[Ny, j] = (2/3)*(((-1)^(j + Ny - 2))/c(j))*(((2*(Ny - 1)^2 + 1)*(1 + x(j)) - 6)/((1 + x(j))^2))
        else
            # evaluate everything else
            diffmat[i, j] = (((-1)^(i + j + 1))/c(j))*((2 - (x(i)*x(j)) - (x(i)^2))/((1 - (x(i)^2))*(x(i) - x(j))^2))
        end
    end

    # re-evaluate the diagonal corners
    diffmat[Ny, Ny] = diffmat[1, 1] = ((Ny - 1)^4 - 1)/15

    return diffmat
end

"""
    Calculate the first order Chebyshev differentiation matrix for a given
    number of Chebyshev discretisation points.

    U. Ehrenstein, R. Peyret (1989), A Chebyshev collocation method for the
    Navier-Stokes equations with application to double-diffusive convection.
"""
function chebdiff(Ny::Int, ::Type{T}=Float64) where {T}
    # initialise matrix
    diffmat = Matrix{T}(undef, Ny, Ny)

    # define anonymous function for c_k coefficient and x location
    c = i -> i == 1 || i == Ny ? 2 : 1
    x = i -> cos(((i - 1)*π)/(Ny - 1))

    # loop over matrix assigning values
    for i in 1:Ny, j in 1:Ny
        if i == j
            # evaluate diagonals
            diffmat[i, i] = -x(i)/(2*(1 - x(j)^2))
        else
            # evaluate everything else
            diffmat[i, j] = (c(i)/c(j))*(((-1)^(i + j))/(x(i) - x(j)))
        end
    end

    # re-evaluate the diagonal corners
    diffmat[1, 1] = (2*(Ny - 1)^2 + 1)/6
    diffmat[Ny, Ny] = -diffmat[1, 1]

    return diffmat
end

"""
    Calculate the quadrature weights for a given Chebyshev grid size, based on
    the Clenshaw-Curtis quadrature method.
"""
chebws(Dy::Matrix{T}) where {T} = append!(inv(Dy[1:(end - 1), 1:(end - 1)])[1, :], [0])


end
