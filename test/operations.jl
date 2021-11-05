@testset "operations" begin
    @testset "Bipolar" begin
        vectors = [[-1, 1, 1, -1],
                   [1, -1, -1, 1],
                   [-1, 1, 1, -1]]

        hdvs = BipolarHDV.(vectors)

        @test (hdvs[1] + hdvs[2] .== [0, 0, 0, 0]) |> all
        @test aggregate(hdvs) == last(hdvs)

        @test (hdvs[1] * hdvs[2] .== [-1, -1, -1, -1]) |> all
        @test bind(hdvs) == BipolarHDV([1, -1, -1, 1])

        hdv = first(hdvs)
        v = first(vectors) |> copy

        @test Π(hdv, 2) == circshift(v, 2)
        @test Π!(hdv, 2) == circshift(v, 2)
        @test resetoffset!(hdv) == v
        @test hdv.v == v 
    end

    @testset "Binary" begin
        vectors = [[true, false, false, true],
                   [true, false, false, true],
                   [false, true, true, false]]

        hdvs = BinaryHDV.(vectors)

        
        @test aggregate(hdvs) == BinaryHDV([true, false, false, true])

        @test (hdvs[1] * hdvs[2] .== [false, false, false, false]) |> all
        @test bind(hdvs[2:3]) == BinaryHDV([true, true, true, true])

        hdv = first(hdvs)
        v = first(vectors) |> copy

        @test Π(hdv, 2) == circshift(v, 2)
        @test Π!(hdv, 2) == circshift(v, 2)
        @test resetoffset!(hdv) == v
        @test hdv.v == v 
    end

    @testset "GradedBipolarHDV" begin
        vectors = [[0, 0, 0, 0.0],
                   [0.8, 0.1, -0.3, 0.2],
                   [-1.0, 1, 1, -1]]

        hdvs = GradedBipolarHDV.(vectors)

        @test hdvs[1] + hdvs[2] ≈ hdvs[2]
        @test aggregate(hdvs) ≈ last(hdvs)

        @test hdvs[1] * hdvs[2] ≈ hdvs[1]
        @test bind(hdvs) ≈ BipolarHDV([0.0, 0.0, 0.0, 0.0])


        hdv = first(hdvs)
        v = first(vectors) |> copy

        @test Π(hdv, 2) == circshift(v, 2)
        @test Π!(hdv, 2) == circshift(v, 2)
        @test resetoffset!(hdv) == v
        @test hdv.v == v 
    end

    @testset "GradedHDV" begin
        vectors = [[0.5, 0.5, 0.5, 0.5],
                   [0.8, 0.1, 0.3, 0.2],
                   [1.0, 0, 1, 0]]

        hdvs = GradedHDV.(vectors)

        @test hdvs[1] + hdvs[2] ≈ hdvs[2]
        @test aggregate(hdvs) ≈ last(hdvs)

        @test hdvs[1] * hdvs[2] ≈ hdvs[1]
        @test bind(hdvs) ≈ hdvs[1]

        hdv = first(hdvs)
        v = first(vectors) |> copy

        @test Π(hdv, 2) == circshift(v, 2)
        @test Π!(hdv, 2) == circshift(v, 2)
        @test resetoffset!(hdv) == v
        @test hdv.v == v 
    end

    @testset "RealHDV" begin
        vectors = [[0, 0, 0, 0.0],
                   [0.8, 0.1, -0.3, 0.2],
                   [-1.0, 1, 1, -1]]

        hdvs = RealHDV.(vectors)

        @test hdvs[1] + hdvs[2] ≈ hdvs[2] / sqrt(2)
        @test aggregate(hdvs) ≈ RealHDV(sum(vectors) / √(3))

        @test hdvs[1] * hdvs[2] ≈ hdvs[1]
        @test bind(hdvs) ≈ hdvs[1] .* hdvs[2] .* hdvs[3]


        hdv = first(hdvs)
        v = first(vectors) |> copy

        @test Π(hdv, 2) == circshift(v, 2)
        @test Π!(hdv, 2) == circshift(v, 2)
        @test resetoffset!(hdv) == v
        @test hdv.v == v 
    end
end