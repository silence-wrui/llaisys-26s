target("llaisys-device-nvidia")
    set_kind("static")
    set_policy("build.cuda.devlink", true)
    add_culdflags("-Xcompiler=-fPIC", {force = true})

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("../src/device/nvidia/*.cu")
    add_cugencodes("native", "compute_75")
    add_links("cudart")

    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end

    on_install(function (target) end)
target_end()


target("llaisys-ops-nvidia")
    set_kind("static")
    set_policy("build.cuda.devlink", true)
    add_culdflags("-Xcompiler=-fPIC", {force = true})

    add_deps("llaisys-tensor")

    set_languages("cxx17")
    set_warnings("all", "error")

    -- CUDA operator sources will be added in later stages.
    add_files("../src/ops/*/nvidia/*.cu")
    add_cugencodes("native", "compute_75")
    add_links("cudart", "cublas")

    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end

    on_install(function (target) end)
target_end()
