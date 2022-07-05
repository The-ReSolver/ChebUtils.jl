# This file defines the necessary linear algebra methods for the ChebDiff type.

LinearAlgebra.lu!(D::ChebDiff) = LinearAlgebra.lu!(D.mat)
