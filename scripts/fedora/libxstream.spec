%global debug_package %{nil}

Name:           libxstream
Version:        0.9.1
Release:        %autorelease
Summary:        OpenCL-accelerated tensor operations built on LIBXS

License:        BSD-3-Clause
URL:            https://github.com/hfp/libxstream
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  bash
BuildRequires:  gcc
BuildRequires:  gcc-gfortran
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
This package contains headers, pkg-config metadata, CMake package files,
the supported header-only source tree, OpenCL kernels, and API documentation
for developing applications that use LIBXSTREAM.

%prep
%autosetup

%build
%make_build GNU=1 STATIC=0 \
    POUTDIR=%{_lib} PPKGDIR=%{_lib}/pkgconfig PCMKDIR=%{_lib}/cmake/%{name}

%install
%make_install PREFIX=%{_prefix} CLEAN=0 STATIC=0 \
    POUTDIR=%{_lib} PPKGDIR=%{_lib}/pkgconfig PCMKDIR=%{_lib}/cmake/%{name}

rm -f %{buildroot}%{_datadir}/%{name}/LICENSE.md

%files
%license LICENSE.md
%doc README.md
%{_libdir}/libxstream.so.*

%files devel
%license LICENSE.md
%doc %{_datadir}/%{name}
%{_includedir}/%{name}/
%{_libdir}/libxstream.so
%{_libdir}/pkgconfig/libxstream*.pc
%{_libdir}/cmake/libxstream/

%changelog
%autochangelog
