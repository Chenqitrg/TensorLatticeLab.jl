Configurations.from_dict(::Type{TransferMatrixConfig}, ::Configurations.OptionField{:which},
                         ::Type{KrylovKit.Selector}, value::AbstractString) = Symbol(value)
Configurations.from_dict(::Type{TransferArnoldi}, ::Configurations.OptionField{:kwargs},
                         ::Type{NamedTuple}, fields::AbstractDict) =
    (; (Symbol(key) => value for (key, value) in fields)...)
