add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

option("mpi")
    set_default(false)
    set_showmenu(true)
    set_description("Enable MPI-based distributed inference on CPU")
option_end()

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    if has_config("mpi") then
        add_defines("ENABLE_MPI")

        on_load(function (target)
            local find_tool = import("lib.detect.find_tool")
            local mpicxx = find_tool("mpicxx") or find_tool("mpic++")
            if not mpicxx then
                raise("MPI enabled, but mpicxx/mpic++ was not found in PATH. Please install openmpi-bin/libopenmpi-dev or mpich/libmpich-dev.")
            end

            local cxxflags = os.iorunv(mpicxx.program, {"--showme:compile"}) or ""
            local ldflags  = os.iorunv(mpicxx.program, {"--showme:link"}) or ""

            cxxflags = cxxflags:gsub("%s+$", "")
            ldflags  = ldflags:gsub("%s+$", "")

            for _, arg in ipairs(os.argv(cxxflags)) do
                if arg:sub(1, 2) == "-I" then
                    target:add("includedirs", arg:sub(3))
                elseif arg:sub(1, 2) == "-D" then
                    target:add("defines", arg:sub(3))
                else
                    target:add("cxflags", arg, {force = true})
                end
            end

            for _, arg in ipairs(os.argv(ldflags)) do
                if arg:sub(1, 2) == "-L" then
                    target:add("linkdirs", arg:sub(3))
                elseif arg:sub(1, 2) == "-l" then
                    target:add("links", arg:sub(3))
                else
                    target:add("shflags", arg, {force = true})
                    target:add("ldflags", arg, {force = true})
                end
            end
        end)
    end

    add_files("src/llaisys/*.cc")
    set_installdir(".")

    after_install(function (target)
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()