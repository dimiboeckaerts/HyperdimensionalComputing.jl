@testset "encoding" begin
    # TODO add more examples and types to test

    hdvs = [GradedHDV(10) for i in 1:5]

    @testset "NGrams" begin

        fourgrams = compute_4_grams(hdvs)

        @test fourgrams.d[2][1][4][3] ≈ Π(hdvs[2], 0) * Π(hdvs[1], 1) * Π(hdvs[4], 2) * Π(hdvs[3], 3)
    end 

    @testset "sequence embedding" begin
        sequence = [2, 3, 5, 1, 3, 4]

        emb = sequence_embedding(sequence, hdvs, 3)

        @test typeof(emb) == eltype(hdvs)

        threegrams = compute_3_grams(hdvs)

        @test sequence_embedding(sequence, threegrams) ≈ emb
    end
end