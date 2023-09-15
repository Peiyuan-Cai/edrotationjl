using LinearAlgebra

function sn(binary)
    """
    It's the sigma_z on ising basis, counting the #up-#down
    input:

    binary::string, the binary of decimal basis index n

    output:

    sn::int, #up-#down
    """
    nup = count('1', binary)
    ndown = count('0', binary)
    return nup - ndown
end

function decimal_to_binary(decimal::Int)
    binary_string = string(decimal, base=2, pad=16)
    return binary_string
end

function u1r(state)
    statecp = copy(state)
    for i in eachindex(state)
        oprbin = decimal_to_binary(i-1)
        s = sn(oprbin)
        statecp[i] *= exp(-1im * π/3 * s)
    end
    return statecp
end

function swap_bits(binary_string::AbstractString)
    permutation=[1,16,11,6,2,13,12,7,3,14,9,8,4,15,10,5]
    #permutation=[1,5,9,13,16,4,8,12,11,15,3,7,6,10,14,2]

    if length(binary_string) != 16
        throw(ArgumentError("Input binary string must have a length of 16."))
    end
    
    if length(permutation) != 16 || length(unique(permutation)) != 16
        throw(ArgumentError("Permutation array must contain unique integers from 0 to 15."))
    end
    
    result = Vector{Char}(undef, 16)
    
    for i in 1:16
        result[i] = binary_string[permutation[i]]
    end
    
    return join(result)
end

function pn(n)
    oprbin = decimal_to_binary(n)
    swpbin = swap_bits(oprbin)
    return parse(Int, swpbin, base=2)
end

function c3r(groundstate)
    u1rtstate = u1r(groundstate)
    res = zeros(ComplexF64,length(groundstate))
    for i in eachindex(res)
        res[pn(i-1)+1] = u1rtstate[i]
    end
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    display(swap_bits("0110101100011010"))
end