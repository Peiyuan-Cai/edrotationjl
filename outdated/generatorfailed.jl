pushfirst!(LOAD_PATH, "~/trans/edrotationjl/")
include("struct_qshlistit.jl")
include("rotation.jl")

using ArgParse
using Arpack
using LinearAlgebra
using SparseArrays
using Plots
using Printf

s0 = [1 0; 0 1]
sx = [0 1; 1 0] / 2
sy = [0 -1im; 1im 0] / 2
sz = [1 0; 0 -1] / 2
sp = sx + 1im*sy
sm = sx - 1im*sy

sprs0 = sparse(s0)
sprsx = sparse(sx)
sprsy = sparse(sy)
sprsz = sparse(sz)
sprsp = sparse(sp)
sprsm = sparse(sm)

function threeOP(ops, site, leng)
    op = sparse(I(1))
    sortops = [ops[i] for i in sortperm(site)]
    site = sort(site)
    for i in 1:2
        if i > 1
            sj = site[i - 1]
        else
            sj = 0
        end
        op = kron(op, kron(sparse(I(2^(site[i] - sj-1))), sortops[i]))
    end
    return kron(op, sparse(I(2^(leng - site[end]))))
end

function hamiltonian(hlist, L, paras=Dict())
    """
    generate hamiltonian
    """
    Jxy = get(paras, "Jxy", 1)
    Jz = get(paras, "Jz", 2)
    Jc = get(paras, "Jc", 2)
    Jcz = get(paras, "Jcz", 1)
    dimh = 2^L
    ham = spzeros(dimh,dimh)
    lst = hlist.hlist
    srtlst = hlist.sortedhlist
    
    # the XXZ terms
    println("Kroning XXZ terms")
    for k in 1:length(lst)
        ham += Jxy * threeOP([sprsx, sprsx], [Int(lst[k][1]), Int(lst[k][2])], L)
        ham += Jxy * threeOP([sprsy, sprsy], [Int(lst[k][1]), Int(lst[k][2])], L)
        ham += Jz * threeOP([sprsz, sprsz], [Int(lst[k][1]), Int(lst[k][2])], L)
    end
    
    # the compass terms and in-out terms
    for k in 1:3
        if k == 1
            ex, ey, ez = 1, 0, 0
        elseif k == 2
            ex, ey, ez = 1/2, √3/2, 0
        elseif k == 3
            ex, ey, ez = -1/2, √3/2, 0
        end
        println("Kroning the $k type bonds")
        for l = 1:length(srtlst[k])
            # compass
            ham += 4*Jc*((ex^2)-0.5) * threeOP([sprsx, sprsx], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            ham += 4*Jc*ex*ey * threeOP([sprsx, sprsy], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            ham += 4*Jc*ey*ex * threeOP([sprsy, sprsx], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            ham += 4*Jc*((ey^2)-0.5) * threeOP([sprsy, sprsy], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            # in-out
            ham += -2*Jcz*ex * threeOP([sprsy, sprsz], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            ham += 2*Jcz*ey * threeOP([sprsx, sprsz], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            ham += -2*Jcz*ex * threeOP([sprsz, sprsy], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            ham += 2*Jcz*ey * threeOP([sprsz, sprsx], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
        end
    end
    
    return ham
end

function hamiltonianalt(hlist, L, paras=Dict())
    """
    generate hamiltonian
    """
    Jxy = get(paras, "Jxy", 1)
    Jz = get(paras, "Jz", 2)
    Jc = get(paras, "Jc", 2)
    Jcz = get(paras, "Jcz", 1)
    dimh = 2^L
    ham = spzeros(dimh,dimh)
    lst = hlist.hlist
    srtlst = hlist.sortedhlist
    
    # the XXZ terms
    println("Kroning XXZ terms")
    for k in 1:length(lst)
        ham += Jxy * threeOP([sprsx, sprsx], [Int(lst[k][1]), Int(lst[k][2])], L)
        ham += Jxy * threeOP([sprsy, sprsy], [Int(lst[k][1]), Int(lst[k][2])], L)
        ham += Jz * threeOP([sprsz, sprsz], [Int(lst[k][1]), Int(lst[k][2])], L)
    end
    
    # the compass terms and in-out terms
    for k in 1:3
        if k == 1
            gam = 1
        elseif k == 2
            gam = exp(1im *2*π/3)
        elseif k == 3
            gam = exp(-1im *2*π/3)
        end
        println("Kroning the $k type bonds")
        for l = 1:length(srtlst[k])
            # compass
            ham += Jc*gam * threeOP([sprsp, sprsp], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            ham += Jc*conj(gam) * threeOP([sprsm, sprsm], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            # in-out
            ham += -1im*Jcz*conj(gam)/2 * threeOP([sprsp, sprsz], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            ham += 1im*Jcz*gam/2 * threeOP([sprsm, sprsz], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            ham += -1im*Jcz*conj(gam)/2 * threeOP([sprsz, sprsp], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
            ham += 1im*Jcz*gam/2 * threeOP([sprsz, sprsm], [Int(srtlst[k][l][1]), Int(srtlst[k][l][2])], L)
        end
    end
    
    return ham
end

function cormn(m,n,state,len)
    smn = threeOP([sprsz,sprsz],[m,n],len)
    return dot(state, smn*state)
end

function cormat(state,len)
    cmat = zeros(len, len)
    for i in 1:len
        for j in 1:len
            if i != j
                cmat[i,j] = real(cormn(i,j,state,len))
            else
                cmat[i,j] = 0.25
            end
        end
    end
    return cmat
end

function main()
    #argparse for command line setups
    parser = ArgParseSettings()
    @add_arg_table parser begin
        "--Jxy"
            arg_type = Float64
            default = 1.317
        "--Jz"
            arg_type = Float64
            default = 1.
        "--Jc"
            arg_type = Float64
            default = 0.
        "--Jcz"
            arg_type = Float64
            default = 0.
    end
    args = parse_args(parser)

    Lx = 4
    Ly = 4
    bcon = "torus"
    L = Lx * Ly
    dimh = 2^L

    hl = Hlist(bcon, Lx, Ly)
    hm = hamiltonianalt(hl, L, Dict("Jxy"=>args["Jxy"], "Jz"=>args["Jz"], "Jc"=>args["Jc"], "Jcz"=>args["Jcz"]))

    println("Diagonalizing")
    uu, vv = eigs(hm, which=:SR)
    println("lowest few energie ", uu)
    println("ground state energy ", uu[1])
    println("E/L ", uu[1] / L)
    gs = vv[:, 1] / norm(vv[:, 1])
    println("should be one ", dot(gs, gs))
    println("check s-equation should be zero ", norm(hm * gs - uu[1] * gs))

    println("rotation part")
    rtgs = c3r(gs)
    println(" ")
    println("<Rg|Rg>=", dot(rtgs, rtgs))
    println("<g|Rg>=", dot(gs, rtgs))
    println("<Rg|H|Rg>=", dot(rtgs, hm*rtgs))
    println("norm of Rg", norm(rtgs))
    println("Rg is not a eigenstate of H if it's not zero: ", norm((hm-dot(rtgs, hm*rtgs)*I(dimh))*rtgs))

    rtrtgs = c3r(rtgs)
    println(" ")
    println("<g|RRg>=", dot(gs,rtrtgs))
    println("<Rg|RRg>=", dot(rtgs, rtrtgs))
    println("<RRg|H|RRg>=", dot(rtrtgs, hm*rtrtgs))
    println("RRg is not a eigenstate of H if it's not zero: ", norm((hm-dot(rtrtgs, hm*rtrtgs)*I(dimh))*rtrtgs))

    rtrtrtgs = c3r(rtrtgs)
    println(" ")
    println("<g|RRRg>=", dot(gs,rtrtrtgs))
    println("<Rg|RRRg>=", dot(rtgs, rtrtrtgs))
    println("<RRg|RRRg>=", dot(rtrtgs, rtrtrtgs))
    println("<RRRg|H|RRRg>=", dot(rtrtrtgs, hm*rtrtrtgs))
    println("RRRg is not a eigenstate of H if it's not zero: ", norm((hm-dot(rtrtrtgs, hm*rtrtrtgs)*I(dimh))*rtrtrtgs))

    println(" ")
    println("<g|HR-RH|g>=", dot(gs, hm*rtgs) - dot(gs, c3r(hm*gs)))
    println("<Rg|HR-RH|Rg>=", dot(rtgs, hm*rtgs) - dot(gs, c3r(hm*rtgs)))
    psi = rand(ComplexF32, dimh)
    psi /= norm(psi)
    println("<psi|HR-RH|psi>=", dot(psi, hm*rtgs) - dot(gs, c3r(hm*psi)))

    println(" ")
    println("Correlations")
    println("cor1 6=", cormn(1,6,gs,L))
    println("cor1 4=", cormn(1,4,gs,L))
    println("cor1 13=", cormn(1,13,gs,L))

    println(" ")
    println("cormat")
    cmt = cormat(gs,L)
    display(cmt)
    heatmap(cmt, c=:blues, xlabel="row", ylabel="col", title="correlation matrix")
    # for i in 1:L
    #     for j in 1:L
    #         annotate!((i - 0.5, j - 0.5, @sprintf("%.4f", cmt[i, j])), textsize= 0.1, color=:white)
    #     end
    # end
    savefig("cormat.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end