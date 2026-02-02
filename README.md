# Scalers.jl

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Julia](https://img.shields.io/badge/julia-1.9%20%7C%201.10%20%7C%201.11-9558B2.svg)](https://julialang.org)

**Scalers.jl** provides MLJ-compatible transformers for feature scaling and normalisation. It implements both linear scaling (MinMaxScaler) and non-linear transformations (QuantileTransformer) that follow the standard [scikit-learn](https://scikit-learn.org/stable/api/sklearn.preprocessing.html) conventions and integrate with the [MLJModelInterface](https://github.com/JuliaAI/MLJModelInterface.jl).

## Overview

Feature scaling is an essential preprocessing step for many machine learning algorithms. `Scalers` provides robust, production-ready transformers that handle edge cases gracefully and integrate seamlessly into MLJ pipelines.

### Available Transformers

#### MinMaxScaler

Linear scaling that transforms features to a specified range (default: [0, 1]):

- Learns minimum and maximum values per feature during fitting
- Applies linear transformation: `x_scaled = (x - min) / (max - min) * (upper - lower) + lower`
- Handles constant columns by mapping to the range minimum
- Supports custom feature-specific ranges

#### QuantileTransformer

Non-linear transformation using empirical cumulative distribution:

- Learns quantiles per feature during fitting (default: 100 quantiles)
- Maps features to uniform distribution over specified range
- Robust to outliers (unlike MinMaxScaler)
- Uses interpolation for out-of-sample values
- Handles ties and non-finite values gracefully

### Key Features

- **MLJ integration**: Full `fit`, `transform`, `inverse_transform` support
- **Type stability**: Efficient type-promoted computations
- **Column preservation**: Maintains exact column ordering from training
- **Robust error handling**: Validates column matching between training and new data
- **Edge case coverage**: Handles constant columns, empty data, and non-finite values

## Installation

```julia
using Pkg
Pkg.add("Scalers")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/Scalers.jl")
```

## Quick Start

### MinMaxScaler Example

```julia
using MLJBase
using Scalers
using DataFrames

# Create sample data
X = DataFrame(
    feature1 = [1.0, 2.0, 3.0, 4.0, 5.0],
    feature2 = [10.0, 20.0, 30.0, 40.0, 50.0]
)

# Fit and transform to [0, 1] range
scaler = MinMaxScaler()
mach = machine(scaler, X)
fit!(mach)

X_scaled = MLJBase.transform(mach, X)
println(X_scaled)

# Inverse transform back to original scale
X_original = MLJBase.inverse_transform(mach, X_scaled)
println(X_original)
```

### QuantileTransformer Example

```julia
using MLJBase
using Scalers
using DataFrames

# Data with outliers
X = DataFrame(
    feature1 = [1.0, 2.0, 3.0, 4.0, 100.0],  # outlier
    feature2 = [10.0, 20.0, 30.0, 40.0, 50.0]
)

# Quantile transformation is robust to outliers
transformer = QuantileTransformer(n_quantiles=10)
mach = machine(transformer, X)
fit!(mach)

X_transformed = transform(mach, X)
println(X_transformed)

# Examine learned quantiles
params = fitted_params(mach)
println("Quantiles for feature1: ", params.quantiles[:feature1])
```

### Pipeline Integration

```julia
using MLJBase
using MLJDecisionTreeInterface
using Scalers

# Create a pipeline with scaling
pipe = Pipeline(
    scaler = MinMaxScaler(),
    classifier = DecisionTreeClassifier()
)

# Use in MLJ workflow
X, y = @load_iris
mach = machine(pipe, X, y)
fit!(mach)
predictions = predict(mach, X)
```

## Licence

This software is distributed under the MIT Licence.
