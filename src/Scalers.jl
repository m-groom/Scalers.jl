module Scalers

using MLJModelInterface
using Tables
using Statistics

include("utilities.jl")
include("main.jl")

export MinMaxScaler, QuantileTransformer

# Common package metadata
const PKG_METADATA = (
    package_name="Scalers",
    package_uuid="857d3a31-ba67-457f-9b14-0a8f313fa218",
    package_url="https://github.com/m-groom/Scalers.jl",
    package_license="MIT",
)

MLJModelInterface.metadata_pkg.(
    (MinMaxScaler, QuantileTransformer);
    PKG_METADATA...,
    is_pure_julia=true,
    is_wrapper=false,
)

end # module Scalers
