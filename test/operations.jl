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

        # add test permutations
    end

    @testset "Binary" begin
        vectors = [[true, false, false, true],
                   [true, false, false, true],
                   [false, true, true, false]]

        hdvs = BinaryHDV.(vectors)

        
        @test aggregate(hdvs) == BinaryHDV([true, false, false, true])

        @test (hdvs[1] * hdvs[2] .== [false, false, false, false]) |> all
        @test bind(hdvs[2:3]) == BinaryHDV([true, true, true, true])

        # add test permutations
    end
end