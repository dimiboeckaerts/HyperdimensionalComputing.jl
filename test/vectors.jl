const n = 10

@testset "vectors" begin
    @testset "BipolarHDV" begin

        hdv = BipolarHDV(n)

        @test length(hdv) == n
        @test eltype(hdv) <: Int
        @test hdv[2] isa Int
        @test all(-1 .≤ hdv .≤ 1)
        @test hdv == BipolarHDV(hdv.v)
        @test similar(hdv) isa BipolarHDV
        @test sum(hdv) ≈ sum(hdv.v)

        hdv_offset = BipolarHDV(hdv.v, 2)
        @test hdv_offset.offset == 2
        @test [hdv[i-2] for i in 1:5] ≈ [hdv_offset[i] for i in 1:5]
        hdv_offset[5] = 0
        @test hdv[3] == 0
    end

    @testset "BinaryHDV" begin

        hdv = BinaryHDV(n)

        @test length(hdv) == n
        @test eltype(hdv) <: Bool
        @test hdv[2] isa Bool
        @test hdv == BinaryHDV(hdv.v)
        @test similar(hdv) isa BinaryHDV
        @test sum(hdv) ≈ sum(hdv.v)

        hdv_offset = BinaryHDV(hdv.v, 2)
        @test hdv_offset.offset == 2
        @test [hdv[i-2] for i in 1:5] ≈ [hdv_offset[i] for i in 1:5]
        hdv_offset[5] = 0
        @test hdv[3] == 0
    end

    @testset "GradedBipolarHDV" begin

        hdv = GradedBipolarHDV(n)

        @test length(hdv) == n
        @test eltype(hdv) <: Real
        @test hdv[2] isa Real
        @test all(-1 .≤ hdv .≤ 1)
        @test hdv == GradedBipolarHDV(hdv.v)
        @test similar(hdv) isa GradedBipolarHDV
        @test sum(hdv) ≈ sum(hdv.v)
        @test eltype(GradedBipolarHDV(Float32, n)) <: Float32

        hdv_offset = GradedBipolarHDV(hdv.v, 2)
        @test hdv_offset.offset == 2
        @test [hdv[i-2] for i in 1:5] ≈ [hdv_offset[i] for i in 1:5]
        hdv_offset[5] = 0
        @test hdv[3] == 0
    end

    @testset "GradedHDV" begin

        hdv = GradedHDV(n)

        @test length(hdv) == n
        @test eltype(hdv) <: Real
        @test hdv[2] isa Real
        @test all(0 .≤ hdv .≤ 1)
        @test hdv == GradedHDV(hdv.v)
        @test similar(hdv) isa GradedHDV
        @test sum(hdv) ≈ sum(hdv.v)
        @test eltype(GradedHDV(Float32, n)) <: Float32

        hdv_offset = GradedHDV(hdv.v, 2)
        @test hdv_offset.offset == 2
        @test [hdv[i-2] for i in 1:5] ≈ [hdv_offset[i] for i in 1:5]
        hdv_offset[5] = 0
        @test hdv[3] == 0
    end


    @testset "RealHDV" begin

        hdv = RealHDV(n)

        @test length(hdv) == n
        @test eltype(hdv) <: Real
        @test hdv[2] isa Real
    
        @test hdv == RealHDV(hdv.v)
        @test similar(hdv) isa RealHDV
        @test sum(hdv) ≈ sum(hdv.v)
        @test eltype(RealHDV(Float32, n)) <: Float32

        hdv_offset = RealHDV(hdv.v, 2)
        @test hdv_offset.offset == 2
        @test [hdv[i-2] for i in 1:5] ≈ [hdv_offset[i] for i in 1:5]
        hdv_offset[5] = 0
        @test hdv[3] == 0
    end
end