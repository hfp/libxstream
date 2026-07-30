Name:           libxstream
Version:        1.0.0
Release:        %autorelease
Summary:        OpenCL-accelerated tensor operations built on LIBXS

License:        BSD-3-Clause
URL:            https://github.com/hfp/libxstream
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  gcc
BuildRequires:  make
BuildRequires:  ocl-icd-devel
BuildRequires:  opencl-headers
BuildRequires:  libxs-devel

%description
LIBXSTREAM is a library for OpenCL-accelerated tensor operations (batched small
matrix multiplications and related numerics).  It builds on top of LIBXS and
targets GPU offload via a portable OpenCL backend.

%package devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}
Requires:       libxs-devel%{?_isa}
Requires:       ocl-icd-devel%{?_isa}

%description devel
This package contains headers, pkg-config metadata, CMake package files, and
OpenCL kernels for developing applications that use LIBXSTREAM.

%package doc
Summary:        Documentation for %{name}
Requires:       %{name} = %{version}-%{release}
BuildArch:      noarch

%description doc
This package contains the API and usage documentation for LIBXSTREAM.

%prep
%autosetup

%build
# SYM=1 retains debuginfo for the debug packages without enabling assertions,
# and E*FLAGS carry the distribution build flags into the Makefile build.
%make_build GNU=1 STATIC=0 SYM=1 \
    ECFLAGS="%{build_cflags}" ELDFLAGS="%{build_ldflags}" \
    POUTDIR=%{_lib} PPKGDIR=%{_lib}/pkgconfig PCMKDIR=%{_lib}/cmake/%{name}

%install
%make_install PREFIX=%{_prefix} CLEAN=0 STATIC=0 SYM=1 \
    ECFLAGS="%{build_cflags}" ELDFLAGS="%{build_ldflags}" \
    POUTDIR=%{_lib} PPKGDIR=%{_lib}/pkgconfig PCMKDIR=%{_lib}/cmake/%{name}

# The license is packaged via %%license from the source tree; drop the
# redundant copy below %%{_docdir} rather than listing the file twice.
rm -f %{buildroot}%{_docdir}/%{name}/LICENSE.md

%files
%license LICENSE.md
%{_libdir}/libxstream.so.*

%files devel
%{_datadir}/%{name}/
%{_includedir}/%{name}/
%{_libdir}/libxstream.so
%{_libdir}/pkgconfig/libxstream*.pc
%{_libdir}/cmake/libxstream/

%files doc
%license LICENSE.md
%doc %{_docdir}/%{name}/

%changelog
%autochangelog
