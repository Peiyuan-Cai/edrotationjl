struct Hlist
    bc::String
    lx::Int
    ly::Int
    hlist::Vector{Any}
    sortedhlist::Vector{Any}
end

function Hlist(bc, lx, ly)
    hlist = Vector{Any}[]
    sortedhlist = Vector{Any}[]

    # hlist generator and sorted
    fullhlistgen!(hlist, sortedhlist, bc, lx, ly)

    return Hlist(bc, lx, ly, hlist, sortedhlist)
end

function fullhlistgen!(hlist, sortedhlist, bc, lx, ly)
    n = lx * ly
    list0 = []  # 0 links
    list1 = []  # pi/3 links
    list2 = []  # 2pi/3 links

    if bc == "cylinder"
        chitsu = reshape(collect(1:n), (lx, ly))
        @show chitsu

        # the 0 links
        for k in 0:n-2
            if (k+1) % lx != 0
                push!(hlist, [k+1, k+2])
                push!(list0, [k+1, k+2])
            end
        end
        for jj in 1:ly
            push!(hlist, [chitsu[lx, jj], chitsu[1, jj]])
            push!(list0, [chitsu[lx, jj], chitsu[1, jj]])
        end

        # the 1 links
        for k in 0:n-lx-1
            if k % lx != lx-1
                push!(hlist, [k+1, k+lx+2])
                push!(list1, [k+1, k+lx+2])
            end
        end
        for jj in 1:ly-1
            push!(hlist, [chitsu[lx, jj], chitsu[1, jj+1]])
            push!(list1, [chitsu[lx, jj], chitsu[1, jj+1]])
        end

        # the 2 links
        for k in 0:n-lx-1
            push!(hlist, [k+1, k+lx+1])
            push!(list2, [k+1, k+lx+1])
        end

        for k in eachindex(hlist)
            if hlist[k][1] > hlist[k][2]
                hlist[k][1], hlist[k][2] = hlist[k][2], hlist[k][1]
            end
        end

        push!(sortedhlist, list0)
        push!(sortedhlist, list1)
        push!(sortedhlist, list2)
    end

    if bc == "torus"
        chitsu = reshape(collect(1:n), (lx, ly))

        # the 0 links
        for k in 0:n-2
            if (k+1) % lx != 0
                push!(hlist, [k+1, k+2])
                push!(list0, [k+1, k+2])
            end
        end
        for jj in 1:ly
            push!(hlist, [chitsu[lx, jj], chitsu[1, jj]])
            push!(list0, [chitsu[lx, jj], chitsu[1, jj]])
        end

        # the 1 links
        for k in 0:n-lx-1
            if k % lx != lx-1
                push!(hlist, [k+1, k+lx+2])
                push!(list1, [k+1, k+lx+2])
            end
        end
        for jj in 1:ly-1
            push!(hlist, [chitsu[lx, jj], chitsu[1, jj+1]])
            push!(list1, [chitsu[lx, jj], chitsu[1, jj+1]])
        end
        for ii in 1:lx-1
            push!(hlist, [chitsu[ii, ly], chitsu[ii+1, 1]])
            push!(list1, [chitsu[ii, ly], chitsu[ii+1, 1]])
        end
        push!(hlist, [chitsu[1, 1], chitsu[lx, ly]])
        push!(list1, [chitsu[1, 1], chitsu[lx, ly]])

        # the 2 links
        for k in 0:n-lx-1
            push!(hlist, [k+1, k+lx+1])
            push!(list2, [k+1, k+lx+1])
        end
        for ii in 1:lx
            push!(hlist, [chitsu[ii, 1], chitsu[ii, ly]])
            push!(list2, [chitsu[ii, 1], chitsu[ii, ly]])
        end
        

        for k in eachindex(hlist)
            if hlist[k][1] > hlist[k][2]
                hlist[k][1], hlist[k][2] = hlist[k][2], hlist[k][1]
            end
        end

        push!(sortedhlist, list0)
        push!(sortedhlist, list1)
        push!(sortedhlist, list2)
    end
end

function printhlist(hlist::Hlist)
    println("The hlist is:")
    println(hlist.hlist)
end

function printsortedhlist(hlist::Hlist)
    println("The 0 links:")
    println(hlist.sortedhlist[1])
    println("The 1 links:")
    println(hlist.sortedhlist[2])
    println("The 2 links:")
    println(hlist.sortedhlist[3])
end

function findlinktype(hlist::Hlist, link)
    lin1 = [link[1], link[2]]
    lin2 = [link[2], link[1]]

    for type in 1:3
        for k in eachindex(hlist.sortedhlist[type])
            if hlist.sortedhlist[type][k] == lin1 || hlist.sortedhlist[type][k] == lin2
                return type-1 #type-1 is the phase coeff
            end
        end
    end

    println("ERROR: Link not found in hlist")
    println("checklist")
    println("para of method [findlinktype]  ----------  inside hlist")
    exit()
end

if abspath(PROGRAM_FILE) == @__FILE__
    hlist = Hlist("torus", 4, 4)
    printhlist(hlist)
    printsortedhlist(hlist)
    println(findlinktype(hlist, [1, 6]))
    println(findlinktype(hlist, [11, 7]))
end
