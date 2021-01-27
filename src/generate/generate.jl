module _Generate

function _print_feature_lines(io::IO,
                              features,
                              features_vector_name::AbstractString = "_features")
    features_vector = deepcopy(collect(features))
    for i in 1:length(features_vector)
        feature = features_vector[i]
        feature = strip(feature)
        feature = strip(feature, ['+', '-'])
        feature = strip(feature)
        feature = replace(feature, "-" => "_")
        feature = replace(feature, "." => "_")
        feature = uppercase(feature)
        features_vector[i] = feature
    end
    unique!(features_vector)
    sort!(features_vector)
    max_length = maximum(length.(features_vector))
    feature_lines = String[]
    for feature in features_vector
        padding_length = max(0, max_length - length(feature))
        padding = repeat(" ", padding_length)
        feature_line = "const $(feature)$(padding) = any(isequal(\"+$(feature)\"),$(padding) $(features_vector_name))"
        push!(feature_lines, feature_line)
    end
    unique!(feature_lines)
    sort!(feature_lines)
    for feature_line in feature_lines
        println(io, feature_line)
    end
    return nothing
end

end # module __Generate
