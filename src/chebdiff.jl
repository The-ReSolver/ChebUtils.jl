# This file contains the definition of Chebyshev differentiation matrices.

struct ChebDiff{T<:Real, N} <: AbstractMatrix{T}
    mat::Matrix{T}

    ChebDiff(mat) = new{eltype(mat), size(mat, 1)}(mat)
end

ChebDiff(mat::AbstractMatrix, ::Type{S}) where {S<:Real} = ChebDiff(S.(mat))
ChebDiff(mat::AbstractMatrix, ::Type{S}) where {S} = Matrix{S}(S.(mat))

Base.size(::ChebDiff{T, N}) where {T, N} = (N, N)
Base.IndexStyle(::Type{<:ChebDiff}) = Base.IndexLinear()
Base.parent(D::ChebDiff) = D.mat
Base.similar(D::ChebDiff{T}, ::Type{S}=T) where {T, S} = ChebDiff(similar(parent(D)), S)
Base.copy(D::ChebDiff) = ChebDiff(copy(parent(D)))

Base.@propagate_inbounds function Base.getindex(D::ChebDiff, I...)
    @boundscheck checkbounds(parent(D), I...)
    @inbounds ret = parent(D)[I...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(D::ChebDiff, v, I...)
    @boundscheck checkbounds(parent(D), I...)
    @inbounds parent(D)[I...] = v
    return v
end

# ~ BROADCASTING ~
# taken from MultiscaleArrays.jl
const ChebDiffStyle = Broadcast.ArrayStyle{ChebDiff}
Base.BroadcastStyle(::Type{<:ChebDiff}) = Broadcast.ArrayStyle{ChebDiff}()

# for broadcasting to construct new objects
Base.similar(bc::Base.Broadcast.Broadcasted{ChebDiffStyle}, ::Type{T}) where {T} = ChebDiff(similar(Array{Float64}, axes(bc)), T)
