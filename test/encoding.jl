@testset "ngrams" begin
    # TODO add more examples and types to test

    hdvs = [GradedHDV(10) for i in 1:5]

    fourgrams = compute_4_grams(hdvs)

    @test fourgrams.d[2][1][4][3] ≈ Π(hdvs[2], 0) * Π(hdvs[1], 1) * Π(hdvs[4], 2) * Π(hdvs[3], 3)

end