# This file contains the definition of Chebyshev differentiation matrices.

struct ChebDiff{T<:Real, N} <: AbstractMatrix{T}
    mat::Matrix{T}

    function ChebDiff(mat::AbstractMatrix{T}, ::Type{T}=Float64) where {T}
        size(mat, 1) == size(mat, 2) || throw(ArgumentError("Matrix must be square"))
        new{T, size(mat, 1)}(T.(mat))
    end
end


Base.size(::ChebDiff{T, N}) where {T, N} = (N, N)
Base.IndexStyle(::Type{<:ChebDiff}) = Base.IndexLinear()
Base.parent(D::ChebDiff) = D.mat
Base.similar(D::ChebDiff{T}, ::Type{S}=T) where {T, S} = ChebDiff(similar(D.mat), S)
Base.copy(D::ChebDiff) = ChebDiff(copy(D.mat))

Base.@propagate_inbounds function Base.getindex(D::ChebDiff, I...)
    @boundscheck checkbounds(parent(D), I...)
    @inbounds ret = parent(D)[I...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(D::ChebDiff, v, I...)
    @boundscheck checkbounds(parent(D), I...)
    @inbounds ret = parent(D)[I...] = v
    return v
end
