# This file contains the Clenshaw-Curtis quadrature weights for a Chebyshev
# grid.

# Original code written by Professor Lloyd Trefethen, Oxford University
# Adapted from code obtained at http://people.maths.ox.ac.uk/trefethen/clencurt.m

function chebws(N::Int)
    # obtain domain information
    N -= 1
    θ = (0:N)π/N

    # intialise vectors
    ws = zeros(N + 1)
    v = ones(N - 1)

    # check if N is even or odd
    if mod(N, 2) == 0
        # compute end weights
        ws[1] = ws[N + 1] = 1/(N^2 - 1)

        # loop over interior points updating v
        for k in 1:(N/2 - 1)
            v .-= 2*cos.(2*k*θ[2:N])/(4*(k^2) - 1)
        end

        v .-= cos.(N*θ[2:N])/(N^2 - 1)
    else
        # compute end points
        ws[1] = ws[N + 1] = 1/N^2

        # loop over interior points updating v
        for k in 1:((N - 1)/2)
            v .-= 2*cos.(2*k*θ[2:N])/(4*(k^2) - 1)
        end
    end

    # assign interior values for weights
    ws[2:N] .= 2*v/N

    return ws
end

chebws_dep(D::ChebDiff{T}) where {T} = append!(inv(D[1:(end - 1), 1:(end - 1)])[1, :], [0])
chebws_dep(N::Int) = chebws_dep(chebdiff(N))