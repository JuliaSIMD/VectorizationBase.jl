using Preferences
using Preferences: @load_preference
using Preferences: @set_preferences!
# using Preferences: @delete_preferences! # TODO: uncomment this line

function _get_override(key)
    return @load_preference(key, default = nothing)
end
function get_override(::Type{T}, key) where {T <: Integer}
    value = parse(T, _get_override(key))::T
    return value
end

function has_override(key)
    value = _get_override(key)
    return !(value isa Nothing)
end

function set_override(key, value)
    s = string(value)
    @set_preferences!(key => s)
    @info("The override has been set. You must restart your Julia session for this change to take effect.")
    return nothing
end

function delete_override(key)
    # TODO: replace this with @delete_preferences!
    @set_preferences!(key => nothing)
    @info("The override has been deleted. You must restart your Julia session for this change to take effect.")
    return nothing
end
