import os
import numpy as np
import pybind11
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "2.0.0"

include_dirs = [
    pybind11.get_include(),
    np.get_include(),
]

# compatibility when run in python_bindings
bindings_dir = "python"
if bindings_dir in os.path.basename(os.getcwd()):
    # crinn package
    source_files_crinn = ["./bindings.cc"]
    # crinn_high package
    source_files_crinn_high = ["./bindings_high.cc"]
    include_dirs.extend(["../", "../third_party/helpa"])
else:
    # crinn package
    source_files_crinn = ["./python/bindings.cc"]
    # crinn_high package
    source_files_crinn_high = ["./python/bindings_high.cc"]
    include_dirs.extend(["./", "./third_party/helpa"])

libraries = []
extra_objects = []

# Define both extensions
ext_modules = [
    Extension(
        "crinn",
        source_files_crinn,
        include_dirs=include_dirs,
        libraries=libraries,
        language="c++",
        extra_objects=extra_objects,
    ),
    Extension(
        "crinn_high",
        source_files_crinn_high,
        include_dirs=include_dirs,
        libraries=libraries,
        language="c++",
        extra_objects=extra_objects,
    ),
]

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    if has_flag(compiler, "-std=c++20"):
        return "-std=c++20"
    else:
        raise RuntimeError(
            "Unsupported compiler -- at least C++20 support " "is needed!"
        )

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        "unix": "-Ofast -lrt -march=native -fpic -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0".split()
    }
    link_opts = {
        "unix": [],
    }
    c_opts["unix"].append("-fopenmp")
    link_opts["unix"].extend(["-fopenmp", "-pthread"])

    def build_extensions(self):
        opts = self.c_opts.get("unix", [])
        opts.append(f'-DVERSION_INFO="{self.distribution.get_version()}"')
        opts.append(cpp_flag(self.compiler))
        if has_flag(self.compiler, "-fvisibility=hidden"):
            opts.append("-fvisibility=hidden")
        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get("unix", []))
        build_ext.build_extensions(self)

setup(
    name="crinn",
    version=__version__,
    description="Graph Library for Approximate Similarity Search",
    author="Zihao Wang",
    long_description="""Graph Library for Approximate Similarity Search""",
    ext_modules=ext_modules,
    install_requires=["numpy"],
    packages=["ann_dataset"],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)