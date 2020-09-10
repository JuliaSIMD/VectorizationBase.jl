

"""
P are the pointers
I contains indexes into `strides` (dynamic) and `X` (static)
X static strides
"""
struct GroupedStridedPointers{P,C,B,R,I,X,O}
    ptrs::P
    strides::X
    offsets::O
end

@inline GroupedStridedPointers{P,C,B,R,I}(ptrs::P, strides::X, offsets::O) = GroupedStridedPointers{P,C,B,R,I,X,O}(ptrs, strides, offsets)

"""
G is a tuple(tuple((A_ind,A's dim),(A_ind,A's dim)), ())
it gives the groups.
"""
@inline function grouped_strided_pointer(A::Tuple{Vararg{<:AbstractArray,N}}, ::Val{G}) where {N,G}
    grouped_strided_pointer(
        map(memory_reference, A),
        map(contiguous_axis, A),
        map(contiguous_batch_size, A),
        map(stride_rank, A),
        map(sdstrides, A),
        map(sdoffsets, A),
        map(dense_dims, A),
        Val{G}()
    )
end

@generated function grouped_strided_pointer(
    ptrs::P, contig_axis::C, batch_sz::B, r::R, x::X, o::O, d::D, ::Val{()}
) where {P,C,B,R,X,O,D}
    # no groups
    N = length(P.parameters)
    q = Expr(:block, Expr(:meta, :inline))
    i = 0
    Ct = Expr(:tuple); Bt = Expr(:tuple); Rt = Expr(:tuple); Xt = Expr(:tuple); Ot = Expr(:tuple); It = Expr(:tuple)
    for n in 1:N
        push!(Ct.args, C.parameters[n].parameters[1])
        push!(Bt.args, B.parameters[n].parameters[1])
        push!(Rt.args, R.parameters[n].parameters[1])
        Xₙ = X.parameters[n];# Oₙ = O.parameters[n];
        Itt = Expr(:tuple)
        if !iszero(length(Xₙ.parameters))
            xₙ = Symbol(:x_,n)
            oₙ = Symbol(:o_,n)
            push!(q.args, Expr(:(=), xₙ, Expr(:ref, :x, n)))
            push!(q.args, Expr(:(=), oₙ, Expr(:ref, :o, n)))
            for j ∈ 1:length(Xₙ.parameters)
                push!(Xt.args, Expr(:ref, xₙ, j))
                push!(Ot.args, Expr(:ref, oₙ, j))
                push!(Itt.args, (i += 1))
            end
        end
        push!(It.args, Itt)
    end
    push!(q.args, :(GroupedStridedPointers{$P,$Ct,$Bt,$Rt,$It}($Xt, $Ot)))
    q
end

function ordered_rank_and_sort(R::NTuple{N,Int}) where {N}
    sp = ntuple(zero, Val{N}())
    r = ntuple(n -> sum(R[n] .≥ R), Val{N}())
    @inbounds for n in 1:N
        sp = Base.setindex(sp, n, r[n])
    end
    (r, sp)::NTuple{2,NTuple{N,Int}}
end

function matching_values(Xₙ, j, Xₚ, k)::Bool
    xₙⱼ = Xₙ[j]
    xₙⱼ === Int && return false
    xₚₖ = Xₚ[k]
    xₚₖ === Int && return false
    xₙⱼ === xₚₖ
end
function check_match(pm, Rₙ, Sₙ, Xₙ, Dₙ, j, Rₚ, Sₚ, Dₚ, Xₚ, k)::Bool
    # if strides match, we're done
    matching_values(Xₙ, j, Xₚ, k) && return true
    Rₙⱼ = Rₙ[j]
    Rₚₖ = Rₚ[k]
    if Rₚₖ > 1 && Rₙₖ > 1
        # Otherwise, we check preceding axis
        nind = Sₙ[Rₙₖ-1]
        pind = Sₚ[Rₚₖ-1]
        # for them being of equal size, 
        if pm[nind, pind] && Dₙ[nind] && Dₚ[pind]
            return check_match(pm, Rₙ, Sₙ, Xₙ, Dₙ, nind, Rₚ, Sₚ, Dₚ, Xₚ, pind)
        end
    # elseif isone(Rₚₖ) && isone(Rₙⱼ)
    end
    false
end

@generated function grouped_strided_pointer(
    ptrs::P, contig_axis::C, batch_sz::B, r::R, x::X, o::O, d::D, ::Val{G}
) where {P,C,B,R,X,O,D,G}
    N = length(P.parameters)
    # We search for stride matches here
    # first we go over the groups, looking for matches
    m = Matrix{Matrix{Bool}}(undef, N, N)
    for g ∈ G
        # each `g` is a tuple of (array ind, paired dim)
        ng = length(g)
        for i ∈ 1:ng
            ai, di = g[i]
            idim = length(Xₙ.parameters[ai])
            for j ∈ i+1:ng
                aj, dj = g[j]
                jdim = length(Xₙ.parameters[aj])
                pm = if isdefined(m, LinearIndices(m)[ai,aj])
                    m[ai, aj]
                elseif ai < aj
                    m[ai,aj] = m[aj,ai] = fill(false, jdim, idim)
                    # m[aj,ai] = fill(false, idim, jdim)
                elseif ai > aj
                    m[ai,aj] = m[aj,ai] = fill(false, idim, jdim)
                    # m[ai,aj] = fill(false, jdim, idim)
                end
                if ai < aj
                    pm[dj, di] = true
                elseif ai > aj
                    pm[di, dj] = true
                end
            end
        end
    end
    # Here we check for static size equivalence info
    # m contains information on matching sizes
    q = Expr(:block, Expr(:meta, :inline))
    i = 0
    Ct = Expr(:tuple); Bt = Expr(:tuple); Rt = Expr(:tuple); Xt = Expr(:tuple); Ot = Expr(:tuple); It = Expr(:tuple)
    for n in 1:N
        push!(Ct.args, C.parameters[n].parameters[1])
        push!(Bt.args, B.parameters[n].parameters[1])
        Rₙ = R.parameters[n].parameters[1]
        push!(Rt.args, Rₙ)
        Xₙ = X.parameters[n].parameters
        Itt = Expr(:tuple)
        ndim = length(Xₙ)
        if ndim > 0
            Rₙ, Sₙ = ordered_rank_and_sort(Rₙ)
            Dₙ = D.parameters[n].parameters[1]
            xₙ = Symbol(:x_,n)
            oₙ = Symbol(:o_,n)
            push!(q.args, Expr(:(=), xₙ, Expr(:ref, :x, n)))
            push!(q.args, Expr(:(=), oₙ, Expr(:ref, :o, n)))
            for j in 1:ndim
                # Here, we now check if we actually need to add info, or if we can find it
                match = false
                Rₙⱼ = (Rₙ[j])::Int
                # nprev_max = isone(Rₙⱼ) ? 0 : n - 1
                nprev_max = n - 1
                for nprev in 1:nprev_max
                    isdefined(m, LinearIndices(m)[n,nprev]) || continue
                    pm = m[n, nprev] # just accessing lower triangle, but matrix is symmetric
                    # pm is ndim_n x ndim_nprev
                    # we need to search back through stride ranks...
                    for k ∈ axes(pm,1)
                        Rₚ, Sₚ = ordered_rank_and_sort(R.parameters[nprev].parameters[1])
                        Dₚ = D.parameters[nprev].parameters[1]
                        Xₚ = X.parameters[nprev].parameters
                        match |= check_match(pm, Rₙ, Sₙ, Xₙ, Dₙ, j, Rₚ, Sₚ, Xₚ, Dₚ, k)
                        if match
                            # must push nprev's ind
                            push!(Itt.args, It.args[nprev].args[k])
                            break
                        end
                    end
                end
                if !match
                    push!(Xt.args, Expr(:ref, xₙ, j))
                    push!(Ot.args, Expr(:ref, oₙ, j))
                    push!(It.args, (i += 1))
                end
            end
        end
        push!(It.args, Itt)
    end    
    push!(q.args, :(GroupedStridedPointers{$P,$Ct,$Bt,$Rt,$It}($Xt, $Ot)))
    q
end

@generated function stridedpointers(gsp::GroupedStridedPointers{P,C,B,R,I,X,O}) where {P,C,B,R,X,O,I}
    t = Expr(:tuple)
    for i ∈ eachindex(I)
        Iᵢ = I[i]
        Nᵢ = length(Iᵢ)
        p = Expr(:ref, :ptrs, i)
        # curly = Expr(:curly, :StridedPointer, )
        x = Expr(:tuple); o = Expr(:tuple)
        for j ∈ Iᵢ
            push!(x.args, Expr(:ref, :strds, j))
            push!(o.args, Expr(:ref, :offs, j))
        end
        push!(t.args, Expr(:call, :stridedpointer, p, :(Contiguous{$(C[i])}()), :(ContiguousBatch{$(B[i])}()), :(StrideRank{$(R[i])}()), x, o))
    end
    Expr(:block, Expr(:meta,:inline), :(ptrs = gsp.ptrs), :(strds = gsp.strides), :(offs = gsp.offsets), t)
end

