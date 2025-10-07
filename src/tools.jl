function get_lensletmap(profiles)
    lensetmap = zeros(Int, 2048, 2048)
    for (i, p) in enumerate(profiles)
        if isnothing(p)
            continue
        end
        lensetmap[p.bbox] .= i
    end
    return lensetmap
end
