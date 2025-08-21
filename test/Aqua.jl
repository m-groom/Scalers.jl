using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(
        Scalers;
        ambiguities=true,
        unbound_args=true,
        undefined_exports=true,
        project_extras=true,
        deps_compat=true,
        persistent_tasks=false,
    )
    Aqua.test_persistent_tasks(Scalers; tmax=120)
end
