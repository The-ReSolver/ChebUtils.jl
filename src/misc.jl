# This file contains a handful of other useful methods based on the ChebDiff
# type.

chebpts(N::Int, ::Type{T}=Float64) where {T} = T.(cos.((0:(N - 1))π/(N - 1)))

chebws(D::ChebDiff{T}) where {T} = append!(inv(D[1:(end - 1), 1:(end - 1)])[1, :], [0])
chebws(N::Int) = chebws(chebdiff(N))
