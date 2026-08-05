%bcond tests 1

%global soversion 1

Name:           libxstream
Version:        1.0.0
Release:        %autorelease
Summary:        OpenCL-accelerated tensor operations built on LIBXS

License:        BSD-3-Clause
URL:            https://github.com/hfp/libxstream
Source0:        https://github.com/hfp/libxstream/releases/download/%{version}/%{name}-%{version}.tar.gz

BuildRequires:  gcc
BuildRequires:  cmake
BuildRequires:  libxs-devel
BuildRequires:  ocl-icd-devel
BuildRequires:  opencl-headers
%if %{with tests}
# Clang is required to compile the kernels in testing
BuildRequires:  clang
%endif

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
%autosetup -p1

%conf
%cmake \
    -DBUILD_TESTING:BOOL=%{with tests} \
    -DLIBXSTREAM_OMP:BOOL=ON \
    -DLIBXSTREAM_INSTALL_HEADER_ONLY:BOOL=OFF

%build
%cmake_build

%install
%cmake_install

%check
%if %{with tests}
%ctest --output-on-failure
%endif

%files
%license LICENSE.md
%{_libdir}/libxstream.so.%{soversion}{,.*}

%files devel
%{_datadir}/%{name}/
%{_includedir}/%{name}/
%{_libdir}/libxstream.so
%{_libdir}/pkgconfig/libxstream*.pc
%{_libdir}/cmake/libxstream/

%files doc
%doc %{_docdir}/%{name}/

%changelog
%autochangelog
