######### Constants ##########

const MMI = MLJModelInterface

const MIDPOINT_PROBABILITY = 0.5

######### Utility Functions ##########

function _validate_column_match(input_col_names, training_features)
    input_cols = Set(input_col_names)
    training_cols = Set(training_features)

    missing_cols = setdiff(training_cols, input_cols)
    extra_cols = setdiff(input_cols, training_cols)

    if !isempty(missing_cols) || !isempty(extra_cols)
        error_msg = "Column mismatch between input and training data. "
        if !isempty(missing_cols)
            error_msg *= "Missing columns: $(collect(missing_cols)). "
        end
        if !isempty(extra_cols)
            error_msg *= "Extra columns: $(collect(extra_cols)). "
        end
        error(error_msg)
    end
end

function _extract_column_vector(table, column_name)
    col_data_abstract = Tables.getcolumn(table, column_name)
    # Avoid collect if already an AbstractVector to reduce allocations
    return if col_data_abstract isa AbstractVector
        col_data_abstract
    else
        collect(col_data_abstract)
    end
end

function _create_feature_mapping(features)
    return Dict(feat => i for (i, feat) in enumerate(features))
end

function _build_named_tuple(column_names, column_vectors)
    # Convert to symbols if needed (handles both Symbol and String column names)
    sym_names =
        column_names isa AbstractVector{Symbol} ? column_names : Symbol.(column_names)
    return NamedTuple{Tuple(sym_names)}(Tuple(column_vectors))
end

function _validate_feature_range(feature_range::Tuple{Float64,Float64})
    if feature_range[1] > feature_range[2]
        return "Upper bound of feature_range ($(feature_range[2])) must be greater than or equal to the lower bound ($(feature_range[1])). Resetting to (0.0, 1.0)."
    else
        return ""
    end
end

# Helper function to get the promoted eltype of a table
function get_promoted_eltype(table)
    @assert Tables.istable(table) "Input must be a Tables.jl compatible table"
    col_names = Tables.columnnames(table)
    # Get all column types and promote them to find common supertype
    col_types = [Tables.columntype(table, name) for name in col_names]
    return promote_type(col_types...)
end
