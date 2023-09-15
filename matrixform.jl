include("generator.jl")
include("rotation.jl")
include("struct_qshlistit.jl")
using SparseArrays
using LinearAlgebra

if abspath(PROGRAM_FILE) == @__FILE__
    let
        Rs = spzeros(ComplexF32,2^16,2^16)
        Rr = spzeros(ComplexF32,2^16,2^16)

        for i in 1:2^16
            Rs[i,i] = exp(-1im*π/3*sn(decimal_to_binary(i-1)))
            Rr[pn(i-1)+1,i] = 1
        end

        Lx = 4
        Ly = 4
        L = Lx*Ly
        bcon = "torus"

        hl = Hlist(bcon, Lx, Ly)
        hm = hamiltonian(hl, L, Dict("Jxy"=>1.317, "Jz"=>1., "Jc"=>0., "Jcz"=>0.))

        println("commutator of Rs and Rr", norm(Rs*Rr-Rr*Rs))
        R = Rr*Rs
        println("commutator of H and R", norm(hm*R-R*hm))
    end
end