# This file tests whether the Chebyshev differenatiation matrices are correct.

@testset "Chebyshev differenatial matrices tests" begin
    # generate random integer in range
    rint = rand(2:50)

    # generate first order differential matrices
    diffmat2 = chebdiff(2)
    diffmat3 = chebdiff(3)
    diffmat4 = chebdiff(4)
    diffmat_randsize = chebdiff(rint)

    # generate the second order differential matrix
    double_diffmat_randsize = chebddiff(rint)

    # first order matrices correct
    @test diffmat2 ≈ [0.5 -0.5; 0.5 -0.5]
    @test diffmat3 ≈ [1.5 -2.0 0.5; 0.5 0.0 -0.5; -0.5 2.0 -1.5]
    @test diffmat4 ≈ [19/6 -4.0 4/3 -0.5; 1.0 -1/3 -1.0 1/3; -1/3 1.0 1/3 -1; 0.5 -4/3 4.0 -19/6]

    # size check for random size matrices
    @test size(diffmat_randsize) == (rint, rint)
    @test size(double_diffmat_randsize) == (rint, rint)

    # second order matrices correct (based off first order)
    @test double_diffmat_randsize ≈ diffmat_randsize*diffmat_randsize

    # chebyshev points
    points = chebpts(rint)
    rand_ind = rand(2:(rint - 1))
    @test size(points) == (rint,)
    @test points[1] == -1.0
    @test points[end] == 1.0
    @test points[rand_ind] == Float64(-cos(π*(rand_ind - 1)/(rint - 1)))
end
