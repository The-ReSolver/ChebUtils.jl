# This file defines how the Linear Algebra multiply method works, providing a
# convenient method of differentiating fields of various dimensions.

# differentiate a vector, x
function LinearAlgebra.mul!(y::AbstractArray{S, 1},
                            D::ChebDiff{T},
                            x::AbstractArray{S, 1}) where {S, T}
    LinearAlgebra.mul!(y, D.mat, x)
end

# differentiate an arbitrary dimension array, x, along the first direction
function LinearAlgebra.mul!(y::AbstractArray{S, ND},
                            D::ChebDiff{T},
                            x::AbstractArray{S, ND}) where {S, ND, T}
    @views @inbounds begin
        for I in CartesianIndices(size(x)[2:end])
            LinearAlgebra.mul!(y[:, I], D, x[:, I])
        end
    end

    return y
end

# FIXME: broken
# differentiate an arbitrary dimension array, x, along a given direction, dir.
function LinearAlgebra.mul!(y::AbstractArray{S, ND},
                            D::ChebDiff{T},
                            x::AbstractArray{S, ND},
                            dir::Int) where {S, ND, T}
    # set up cartesian indexes around differentiation direction
    Rpre = CartesianIndices(size(x)[1:(dir - 1)])
    Rpost = CartesianIndices(size(x)[(dir + 1):end])

    @views @inbounds begin
        for Ipost in Rpost, Ipre in Rpre
            LinearAlgebra.mul!(y[Ipre, :, Ipost], D, x[Ipre, :, Ipost])
        end
    end

    return y
end
