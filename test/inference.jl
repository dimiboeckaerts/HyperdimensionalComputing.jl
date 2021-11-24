@testset "inference" begin
    
    @testset "BinaryHDV" begin
        x = BinaryHDV([true, false, true, true])
        y = BinaryHDV([false, false, false, true])

        @test similarity(x, y) ≈ 1 / 3 ≈ jacc_sim(x.v, y.v)

        Π!(x, 2)
        @test similarity(x, y) ≈ jacc_sim(circshift(x.v, 2), y.v)

    end

    @testset "GradedHDV" begin
        x = GradedHDV([0.1, 0.4, 0.6, .8])
        y = GradedHDV([0.9, 0.8, 0.1, 0.3])

        @test similarity(x, y) ≈ jacc_sim(x.v, y.v)

        Π!(x, 2)
        @test similarity(x, y) ≈ jacc_sim(circshift(x.v, 2), y.v)
    end

    @testset "BipolarHDV" begin
        x = BipolarHDV([1, -1, 0, -1])
        y = BipolarHDV([-1, 0, 1, -1])

        @test similarity(x, y) ≈ cos_sim(x.v, y.v)

        Π!(x, 2)
        @test similarity(x, y) ≈ cos_sim(circshift(x.v, 2), y.v)
    end

    @testset "GradedBipolarHDV" begin
        x = GradedBipolarHDV([0.1, -0.4, 0.6, .8])
        y = GradedBipolarHDV([0.9, 0.8, -0.1, -0.3])

        @test similarity(x, y) ≈ cos_sim(x.v, y.v)

        Π!(x, 2)
        @test similarity(x, y) ≈ cos_sim(circshift(x.v, 2), y.v)
    end

    @testset "RealHDV" begin
        x = RealHDV([0.1, -0.4, 0.6, .8])
        y = RealHDV([0.9, 0.8, -0.1, -0.3])

        @test similarity(x, y) ≈ cos_sim(x.v, y.v)

        Π!(x, 2)
        @test similarity(x, y) ≈ cos_sim(circshift(x.v, 2), y.v)
    end
end