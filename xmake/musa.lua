local musa_home = os.getenv("MUSA_HOME") or "/usr/local/musa"
local mcc = path.join(musa_home, "bin", "mcc")

rule("musa.build")
    set_extensions(".mu")

    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        local objectfile = target:objectfile(sourcefile)
        local objectdir = path.directory(objectfile)

        local args = {
            "--musa-path=" .. musa_home,
            "--offload-arch=mp_31",
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-I" .. path.join(os.projectdir(), "include"),
            "-I" .. path.join(musa_home, "include"),
            "-c",
            sourcefile,
            "-o",
            objectfile
        }

        batchcmds:show_progress(
            opt.progress,
            "${color.build.object}compiling.musa %s",
            sourcefile
        )
        batchcmds:mkdir(objectdir)
        batchcmds:vrunv(mcc, args)

        table.insert(target:objectfiles(), objectfile)

        batchcmds:add_depfiles(sourcefile)
        batchcmds:set_depmtime(os.mtime(objectfile))
        batchcmds:set_depcache(target:dependfile(objectfile))
    end)
rule_end()

target("llaisys-device-musa")
    set_kind("static")
    add_rules("musa.build")

    add_files("../src/device/musa/*.mu")

    add_includedirs(
        path.join(musa_home, "include"),
        {public = true}
    )
    add_linkdirs(
        path.join(musa_home, "lib"),
        {public = true}
    )
    add_syslinks("musart", {public = true})

    on_install(function (target) end)
target_end()