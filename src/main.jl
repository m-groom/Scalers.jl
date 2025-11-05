######### MinMaxScaler ##########

mutable struct MinMaxScaler <: MMI.Unsupervised
    feature_range::Tuple{Float64,Float64}
end

# Keyword constructor
function MinMaxScaler(; feature_range=(0.0, 1.0))
    transformer = MinMaxScaler(feature_range)
    message = MMI.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MMI.clean!(transformer::MinMaxScaler)
    err = _validate_feature_range(transformer.feature_range)
    if !isempty(err)
        transformer.feature_range = (0.0, 1.0)
    end
    return err
end

# Fit method: learns min and max for each feature
function MMI.fit(transformer::MinMaxScaler, verbosity::Int, X)
    # X is assumed to be a Tables.jl compatible table.
    col_names = Tables.columnnames(X)
    # Get promoted element type from all columns
    T = get_promoted_eltype(X)
    # Pre-allocate result vectors with known size
    all_mins = Vector{T}(undef, length(col_names))
    all_maxs = Vector{T}(undef, length(col_names))

    for (col_idx, name) in enumerate(col_names)
        col_data = Tables.getcolumn(X, name)
        # Convert to an iterable collection if it's not already one (e.g. a generator) and
        # ensure elements are numbers.
        col_iterable = collect(T, col_data)
        if isempty(col_iterable)
            # Handle empty columns: use NaN
            all_mins[col_idx] = NaN
            all_maxs[col_idx] = NaN
        else
            all_mins[col_idx] = minimum(col_iterable)
            all_maxs[col_idx] = maximum(col_iterable)
        end
    end

    fitresult = (mins=all_mins, maxs=all_maxs, features=col_names)
    cache = nothing # No cache needed
    report = nothing

    return fitresult, cache, report
end

# transform method: applies the scaling
function MMI.transform(transformer::MinMaxScaler, fitresult, Xnew)
    col_names = Tables.columnnames(Xnew)
    data_mins = fitresult.mins
    data_maxs = fitresult.maxs
    features = fitresult.features

    # Validate that input columns exactly match training columns
    _validate_column_match(col_names, features)

    # Create mapping from feature name to index in training data
    feature_to_idx = _create_feature_mapping(features)

    # Get promoted element type from input table
    T = get_promoted_eltype(Xnew)
    f_min, f_max = transformer.feature_range
    f_scale = f_max - f_min

    # Pre-allocate result vector with known size
    scaled_columns = Vector{AbstractVector{T}}(undef, length(col_names))

    for (col_idx, name) in enumerate(col_names)
        col_vector = _extract_column_vector(Xnew, name)

        # Use feature name to get correct min/max values
        feature_idx = feature_to_idx[name]
        current_data_min = data_mins[feature_idx]
        current_data_max = data_maxs[feature_idx]
        data_range = current_data_max - current_data_min

        scaled_col_vector = similar(col_vector, T)

        if data_range == 0.0
            # If data column is constant, map all values to f_min
            scaled_col_vector .= f_min
        else
            inv_data_range = 1.0 / data_range
            @inbounds @simd for i in eachindex(col_vector)
                # Standardise to [0,1] then scale to feature_range
                scaled_col_vector[i] =
                    (col_vector[i] - current_data_min) * inv_data_range * f_scale + f_min
            end
        end

        scaled_columns[col_idx] = scaled_col_vector
    end

    return _build_named_tuple(col_names, scaled_columns)
end

# inverse_transform method: reverses the scaling
function MMI.inverse_transform(transformer::MinMaxScaler, fitresult, Xscaled)
    col_names = Tables.columnnames(Xscaled)
    data_mins = fitresult.mins
    data_maxs = fitresult.maxs
    features = fitresult.features

    # Validate that input columns exactly match training columns
    _validate_column_match(col_names, features)

    # Create mapping from feature name to index in training data
    feature_to_idx = _create_feature_mapping(features)

    # Get promoted element type from input table
    T = get_promoted_eltype(Xscaled)
    f_min, f_max = transformer.feature_range
    f_scale = f_max - f_min

    # Pre-allocate result vector with known size
    restored_columns = Vector{AbstractVector{T}}(undef, length(col_names))

    for (col_idx, name) in enumerate(col_names)
        scaled_col_vector = _extract_column_vector(Xscaled, name)

        # Use feature name to get correct min/max values
        feature_idx = feature_to_idx[name]
        current_data_min = data_mins[feature_idx]
        current_data_max = data_maxs[feature_idx]
        data_range = current_data_max - current_data_min

        restored_col_vector = similar(scaled_col_vector, T)
        if data_range == 0.0
            # If original data column was constant, all values should be current_data_min.
            restored_col_vector .= current_data_min
        elseif f_scale == 0.0
            # Original data had a range, but it was scaled to a single point (f_min).
            # All scaled values should ideally be f_min. The unscaled value (0-1 range) is 0
            restored_col_vector .= current_data_min # restore to current_data_min
        else
            # Both data_range and f_scale are non-zero.
            inv_f_scale = 1.0 / f_scale
            @inbounds @simd for i in eachindex(scaled_col_vector)
                # Ensure input to Float64 conversion if elements are not already floats
                val_01 = (scaled_col_vector[i] - f_min) * inv_f_scale
                restored_col_vector[i] = val_01 * data_range + current_data_min
            end
        end
        restored_columns[col_idx] = restored_col_vector
    end

    return _build_named_tuple(col_names, restored_columns)
end

# Fitted parameters
function MMI.fitted_params(::MinMaxScaler, fitresult)
    return (min_values_per_feature=fitresult.mins, max_values_per_feature=fitresult.maxs)
end

# MLJ Traits
MMI.metadata_model(
    MinMaxScaler;
    input_scitype=MMI.Table(MMI.Continuous),
    output_scitype=MMI.Table(MMI.Continuous),
    human_name="Min-Max Scaler",
    load_path="Scalers.MinMaxScaler",
)

######### QuantileTransformer ##########

mutable struct QuantileTransformer <: MMI.Unsupervised
    feature_range::Tuple{Float64,Float64}
    n_quantiles::Int
end

# Keyword constructor
function QuantileTransformer(; feature_range=(0.0, 1.0), n_quantiles=1000)
    nq = Int(n_quantiles)
    transformer = QuantileTransformer(feature_range, nq)
    message = MMI.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MMI.clean!(transformer::QuantileTransformer)
    err = _validate_feature_range(transformer.feature_range)
    if !isempty(err)
        transformer.feature_range = (0.0, 1.0)
    end
    if transformer.n_quantiles < 1
        err *= " n_quantiles ($(transformer.n_quantiles)) must be at least 1. Resetting to 1000."
        transformer.n_quantiles = 1000
    end

    return err
end

function MMI.fit(transformer::QuantileTransformer, verbosity::Int, X)
    col_names = Tables.columnnames(X)
    # Get promoted element type from all columns
    T = get_promoted_eltype(X)
    # Pre-allocate result vectors with known size
    quantiles_per_column = Vector{Vector{T}}(undef, length(col_names))
    probabilities_per_column = Vector{Vector{T}}(undef, length(col_names))
    constant_mask = BitVector(undef, length(col_names))
    constant_values = Vector{T}(undef, length(col_names))

    for (col_idx, name) in enumerate(col_names)
        col_data = Tables.getcolumn(X, name)
        col_array = _collect_numeric_column(col_data, T)

        if any(isinf, col_array)
            error("Feature $(name) contains ±Inf values which QuantileTransformer cannot handle.")
        end

        finite_mask = .!isnan.(col_array)
        numeric_col_data = col_array[finite_mask]

        if isempty(numeric_col_data)
            quantiles_per_column[col_idx] = T[]
            probabilities_per_column[col_idx] = T[]
            constant_mask[col_idx] = false
            constant_values[col_idx] = T(NaN)
            continue
        end

        quantiles, probabilities = _compute_quantile_grid(numeric_col_data, transformer.n_quantiles)
        quantiles_per_column[col_idx] = quantiles
        probabilities_per_column[col_idx] = probabilities

        is_const = _is_constant_quantile_grid(quantiles)
        constant_mask[col_idx] = is_const
        constant_values[col_idx] = is_const ? quantiles[1] : T(NaN)
    end

    fitresult = (
        quantiles_list=quantiles_per_column,
        probabilities_list=probabilities_per_column,
        constant_mask=constant_mask,
        constant_values=constant_values,
        features=col_names,
    )
    cache = nothing
    report = nothing

    return fitresult, cache, report
end

function MMI.transform(transformer::QuantileTransformer, fitresult, Xnew)
    Xnew_col_names = Tables.columnnames(Xnew)

    # Validate that input columns exactly match training columns
    _validate_column_match(Xnew_col_names, fitresult.features)

    # Create mapping from feature name to index in training data
    feature_to_idx = _create_feature_mapping(fitresult.features)

    # Get promoted element type from input table
    T = get_promoted_eltype(Xnew)
    min_range, max_range = transformer.feature_range
    range_span = max_range - min_range

    # Pre-allocate result vector with known size
    transformed_cols = Vector{AbstractVector{T}}(undef, length(Xnew_col_names))

    for (col_idx, name) in enumerate(Xnew_col_names)
        col_vector = _extract_column_vector(Xnew, name)
        col_values = _collect_numeric_column(col_vector, T)

        # Use feature name to get correct quantiles
        feature_idx = feature_to_idx[name]
        current_quantiles = fitresult.quantiles_list[feature_idx]
        current_probabilities = fitresult.probabilities_list[feature_idx]
        is_constant = fitresult.constant_mask[feature_idx]

        new_col = Vector{T}(undef, length(col_values))
        @inbounds for i in eachindex(col_values)
            val = col_values[i]
            if isnan(val)
                new_col[i] = T(val)
                continue
            elseif !isfinite(val)
                error(
                    "Input to QuantileTransformer contains ±Inf values which are not supported."
                )
            end

            probability = if isempty(current_quantiles)
                MIDPOINT_PROBABILITY
            elseif is_constant
                MIDPOINT_PROBABILITY
            else
                _value_to_probability(val, current_quantiles, current_probabilities)
            end

            scaled_val = range_span == 0.0 ? min_range : probability * range_span + min_range
            new_col[i] = T(scaled_val)
        end

        transformed_cols[col_idx] = new_col
    end

    # Reconstruct the table with the original column names from fitting
    output_col_names = fitresult.features
    if length(transformed_cols) != length(output_col_names)
        error(
            "Internal error: Number of transformed columns does not match number of " *
            "fitted column names.",
        )
    end

    return _build_named_tuple(output_col_names, transformed_cols)
end

function MMI.inverse_transform(transformer::QuantileTransformer, fitresult, Xtransformed)
    Xtransformed_col_names = Tables.columnnames(Xtransformed)

    # Validate that input columns exactly match training columns
    _validate_column_match(Xtransformed_col_names, fitresult.features)

    # Create mapping from feature name to index in training data
    feature_to_idx = _create_feature_mapping(fitresult.features)

    # Get promoted element type from input table
    T = get_promoted_eltype(Xtransformed)
    min_range, max_range = transformer.feature_range
    range_span = max_range - min_range
    # Handle range_span == 0 separately to avoid division by zero with inv_range_span
    inv_range_span = range_span == 0.0 ? 0.0 : 1.0 / range_span # Will be used if range_span != 0

    # Pre-allocate result vector with known size
    original_cols = Vector{AbstractVector{T}}(undef, length(Xtransformed_col_names))

    for (col_idx, name) in enumerate(Xtransformed_col_names)
        col_vector = _extract_column_vector(Xtransformed, name)
        col_values = _collect_numeric_column(col_vector, T)

        # Use feature name to get correct quantiles
        feature_idx = feature_to_idx[name]
        current_quantiles = fitresult.quantiles_list[feature_idx]
        current_probabilities = fitresult.probabilities_list[feature_idx]
        is_constant = fitresult.constant_mask[feature_idx]
        constant_value = fitresult.constant_values[feature_idx]

        new_col = Vector{T}(undef, length(col_values))
        @inbounds for i in eachindex(col_values)
            s_val = col_values[i]
            if isnan(s_val)
                new_col[i] = T(s_val)
                continue
            elseif !isfinite(s_val)
                error(
                    "Input to QuantileTransformer.inverse_transform contains ±Inf values which are not supported."
                )
            end

            probability = if range_span == 0.0
                MIDPOINT_PROBABILITY
            else
                clamp((s_val - min_range) * inv_range_span, zero(T), one(T))
            end

            restored = if isempty(current_quantiles)
                T(NaN)
            elseif is_constant
                constant_value
            else
                _probability_to_value(probability, current_quantiles, current_probabilities)
            end

            new_col[i] = T(restored)
        end
        original_cols[col_idx] = new_col
    end

    output_col_names = fitresult.features
    if length(original_cols) != length(output_col_names)
        error(
            "Internal error: Number of inverse_transformed columns " *
            "does not match number of fitted column names.",
        )
    end

    return _build_named_tuple(output_col_names, original_cols)
end

# Helpers
function _collect_numeric_column(column_data, T::Type{<:AbstractFloat})
    col_iterable = column_data isa AbstractVector ? column_data : collect(column_data)
    result = Vector{T}(undef, length(col_iterable))
    @inbounds for (idx, value) in enumerate(col_iterable)
        if value === missing
            result[idx] = T(NaN)
        else
            result[idx] = T(value)
        end
    end
    return result
end

function _compute_quantile_grid(values::Vector{T}, n_quantiles::Int) where {T<:AbstractFloat}
    sorted_vals = sort(values)
    n_samples = length(sorted_vals)
    n_quantiles_eff = max(2, min(n_quantiles, n_samples))
    probabilities = collect(range(zero(T), one(T); length=n_quantiles_eff))
    quantiles = Vector{T}(undef, n_quantiles_eff)

    if n_samples == 1
        fill!(quantiles, sorted_vals[1])
        return quantiles, probabilities
    end

    @inbounds for (idx, p) in enumerate(probabilities)
        if p <= zero(T)
            quantiles[idx] = sorted_vals[1]
        elseif p >= one(T)
            quantiles[idx] = sorted_vals[end]
        else
            position = (n_samples - 1) * p + 1
            lower_idx = floor(Int, position)
            upper_idx = ceil(Int, position)
            weight = position - lower_idx
            lower_val = sorted_vals[lower_idx]
            upper_val = sorted_vals[upper_idx]
            quantiles[idx] = (1 - weight) * lower_val + weight * upper_val
        end
    end

    return quantiles, probabilities
end

function _is_constant_quantile_grid(quantiles::Vector{T}) where {T<:AbstractFloat}
    isempty(quantiles) && return false
    q_min = minimum(quantiles)
    q_max = maximum(quantiles)
    return isapprox(q_min, q_max; atol=eps(T) * max(one(T), abs(q_min)))
end

function _value_to_probability(
    val::T, quantiles::Vector{T}, probabilities::Vector{T}
) where {T<:AbstractFloat}
    if val <= quantiles[1]
        return zero(T)
    elseif val >= quantiles[end]
        return one(T)
    end

    idx = searchsortedlast(quantiles, val)
    idx = clamp(idx, 1, length(quantiles) - 1)
    upper_idx = idx + 1
    q_low = quantiles[idx]
    q_high = quantiles[upper_idx]
    p_low = probabilities[idx]
    p_high = probabilities[upper_idx]

    if q_high == q_low
        return (p_low + p_high) / 2
    else
        t = (val - q_low) / (q_high - q_low)
        return p_low + t * (p_high - p_low)
    end
end

function _probability_to_value(
    probability::T, quantiles::Vector{T}, probabilities::Vector{T}
) where {T<:AbstractFloat}
    if probability <= probabilities[1]
        return quantiles[1]
    elseif probability >= probabilities[end]
        return quantiles[end]
    end

    idx = searchsortedlast(probabilities, probability)
    idx = clamp(idx, 1, length(probabilities) - 1)
    upper_idx = idx + 1
    p_low = probabilities[idx]
    p_high = probabilities[upper_idx]
    q_low = quantiles[idx]
    q_high = quantiles[upper_idx]

    if p_high == p_low
        return T(0.5) * (q_low + q_high)
    else
        t = (probability - p_low) / (p_high - p_low)
        return q_low + t * (q_high - q_low)
    end
end

# Fitted parameters
function MMI.fitted_params(::QuantileTransformer, fitresult)
    return (
        quantiles_list=fitresult.quantiles_list,
        probabilities_list=fitresult.probabilities_list,
        constant_mask=fitresult.constant_mask,
        constant_values=fitresult.constant_values,
    )
end

# MLJ traits
MMI.metadata_model(
    QuantileTransformer;
    input_scitype=MMI.Table(MMI.Continuous),
    output_scitype=MMI.Table(MMI.Continuous),
    human_name="Quantile Transformer",
    load_path="Scalers.QuantileTransformer",
)

######### Documentation ##########

"""
$(MMI.doc_header(MinMaxScaler))

Use this model to scale features to a given range, defaulting to [0, 1]. Each feature
is scaled independently using a linear transformation based on the minimum and maximum
values observed during fitting. The rescalings applied by this transformer to new data
are always those learned during the training phase. The behaviour of this model is similar
to that of the `MinMaxScaler` in the `sklearn.preprocessing` Python package.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any Tables.jl compatible table or any abstract vector with
  `Continuous` element scitype (any abstract float vector). Only
  features in a table with `Continuous` scitype can be scaled;
  check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `feature_range::Tuple{Float64, Float64}`: The desired range for the transformed data.
  Defaults to `(0.0, 1.0)`.

- `n_quantiles::Int`: The number of points used to approximate the ECDF. Defaults to
  `1000`, is capped by the number of observed samples, and is enforced to be at least 2.


# Operations

- `transform(mach, Xnew)`: return `Xnew` with features scaled to the specified
  `feature_range` according to the min/max values learned during fitting of `mach`.
  The transformation formula is: `X_scaled = (X - data_min) / (data_max - data_min) * (range_max - range_min) + range_min`.
  If a feature has constant values (data_max == data_min), all values are mapped to `range_min`.

- `inverse_transform(mach, Z)`: apply the inverse transformation to `Z`, mapping
  values from `feature_range` back to the original feature domain using the
  min/max values learned during `fit`.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `min_values_per_feature` - the minimum values for each feature column learned during fitting

- `max_values_per_feature` - the maximum values for each feature column learned during fitting


# Examples

```
using MLJ

# Create example data with different scales
X = (a = [1.0, 2.0, 3.0, 4.0, 5.0],
     b = [5.0, 4.0, 3.0, 2.0, 1.0],
     c = [10.0, 20.0, 30.0, 40.0, 50.0])

julia> schema(X)
┌───────┬────────────┬─────────┐
│ names │ scitypes   │ types   │
├───────┼────────────┼─────────┤
│ a     │ Continuous │ Float64 │
│ b     │ Continuous │ Float64 │
│ c     │ Continuous │ Float64 │
└───────┴────────────┴─────────┘

# Default scaling to [0, 1]
scaler = MinMaxScaler()
mach = machine(scaler, X)
fit!(mach)

julia> X_transformed = transform(mach, X)
(a = [0.0, 0.25, 0.5, 0.75, 1.0],
 b = [1.0, 0.75, 0.5, 0.25, 0.0],
 c = [0.0, 0.25, 0.5, 0.75, 1.0],)

# Custom feature range
scaler2 = MinMaxScaler(feature_range=(-1.0, 1.0))
mach2 = machine(scaler2, X)
fit!(mach2)

julia> transform(mach2, X)
(a = [-1.0, -0.5, 0.0, 0.5, 1.0],
 b = [1.0, 0.5, 0.0, -0.5, -1.0],
 c = [-1.0, -0.5, 0.0, 0.5, 1.0],)

# Handling constant columns
X_const = (a = [1.0, 1.0, 1.0], b = [2.0, 3.0, 4.0])
mach_const = fit!(machine(scaler, X_const))

julia> transform(mach_const, X_const)
(a = [0.0, 0.0, 0.0],              # constant column mapped to range_min
 b = [0.0, 0.5, 1.0],)             # regular scaling

# Perfect inverse transformation
julia> X_restored = inverse_transform(mach, X_transformed)
(a = [1.0, 2.0, 3.0, 4.0, 5.0],
 b = [5.0, 4.0, 3.0, 2.0, 1.0],
 c = [10.0, 20.0, 30.0, 40.0, 50.0],)

# View fitted parameters
julia> fitted_params(mach)
(min_values_per_feature = [1.0, 1.0, 10.0],
 max_values_per_feature = [5.0, 5.0, 50.0],)
```

See also [`QuantileTransformer`](@ref).
"""
MinMaxScaler

"""
$(MMI.doc_header(QuantileTransformer))

Use this model to transform features to be uniformly distributed over a given range,
defaulting to [0, 1]. This transformation maps each feature to a uniform distribution
by calculating the empirical cumulative distribution function (ECDF) of the training
data using rank-based percentiles. The rescalings applied by this transformer to new
data are always those learned during the training phase. The behaviour of this model
is similar to that of the `QuantileTransformer` in the `sklearn.preprocessing` Python
package.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any Tables.jl compatible table or any abstract vector with
  `Continuous` element scitype (any abstract float vector). Only
  features in a table with `Continuous` scitype can be transformed;
  check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `feature_range::Tuple{Float64, Float64}`: The desired range for the transformed data.
  Defaults to `(0.0, 1.0)`.

- `n_quantiles::Int`: The number of points used to approximate the ECDF. Defaults to
  `1000`, is capped by the number of observed samples, and is enforced to be at least 2.


# Operations

- `transform(mach, Xnew)`: return `Xnew` with features transformed to a uniform
  distribution over the specified `feature_range` according to the quantile grid
  learned during fitting of `mach`. Values below/above the fitted range are mapped
  to the bounds of `feature_range`, constant training features map to the midpoint
  of that range, columns with no finite training values map to that same midpoint,
  `NaN` values are propagated, and ±`Inf` inputs raise an error.

- `inverse_transform(mach, Z)`: apply the inverse transformation to `Z`, mapping
  values from `feature_range` back to the original feature domain using linear
  interpolation between the stored quantile grid. Constant features revert to the
  fitted constant value, columns with no finite training values yield `NaN`, `NaN`
  values are propagated, and ±`Inf` inputs raise an error.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `quantiles_list` - vector of quantile arrays, one for each feature column

- `probabilities_list` - vector of probability grids paired with each quantile array

- `constant_mask` - boolean vector indicating which feature columns were constant during fitting

- `constant_values` - vector storing the fitted constant value for constant feature columns (NaN otherwise)


# Examples

```
using MLJ

# Create example data with different patterns
X = (a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
     b = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
     c = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0])

julia> schema(X)
┌───────┬────────────┬─────────┐
│ names │ scitypes   │ types   │
├───────┼────────────┼─────────┤
│ a     │ Continuous │ Float64 │
│ b     │ Continuous │ Float64 │
│ c     │ Continuous │ Float64 │
└───────┴────────────┴─────────┘

# Default transformation to [0, 1]
transformer = QuantileTransformer()
mach = machine(transformer, X)
fit!(mach)

julia> X_transformed = transform(mach, X)
(a = [0.0, 0.1111111111111111, 0.2222222222222222, 0.3333333333333333, 0.4444444444444444, 0.5555555555555556, 0.6666666666666666, 0.7777777777777777, 0.8888888888888888, 1.0],
 b = [1.0, 0.8888888888888888, 0.7777777777777777, 0.6666666666666666, 0.5555555555555556, 0.4444444444444444, 0.3333333333333333, 0.2222222222222222, 0.1111111111111111, 0.0],
 c = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],)

# Custom feature range
transformer2 = QuantileTransformer(feature_range=(-1.0, 1.0))
mach2 = machine(transformer2, X)
fit!(mach2)

julia> transform(mach2, X)
(a = [-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333326, 0.5555555555555554, 0.7777777777777777, 1.0],
 b = [1.0, 0.7777777777777777, 0.5555555555555554, 0.33333333333333326, 0.11111111111111116, -0.11111111111111116, -0.33333333333333337, -0.5555555555555556, -0.7777777777777778, -1.0],
 c = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],)

# Out-of-sample data handling
X_new = (a = [0.0, 5.5, 11.0],
         b = [12.0, 5.5, -1.0],
         c = [1.0, 7.0, 10.0])

julia> transform(mach, X_new)
(a = [0.0, 0.5, 1.0],
 b = [1.0, 0.5, 0.0],
 c = [0.0, 0.7, 1.0],)

# Inverse transformation
julia> X_restored = inverse_transform(mach, X_transformed)
(a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.999999999999999, 9.0, 10.0],
 b = [10.0, 9.0, 7.999999999999999, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
 c = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0],)
```

See also [`MinMaxScaler`](@ref).
"""
QuantileTransformer
