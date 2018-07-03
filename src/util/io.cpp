
#include "util/io.h"

#include <cassert>
#include <stdexcept>

namespace {
hid_t enforce(hid_t id)
{
	if (id < 0)
		throw std::runtime_error("HDF5 error");
	return id;
}
} // namespace

DataSet::DataSet(hid_t id) : id(id)
{
	if (id <= 0)
		return;
	auto space = enforce(H5Dget_space(id));
	size = H5Sget_simple_extent_npoints(space);
	H5Sclose(space);
}

DataSet::~DataSet()
{
	if (id > 0)
		H5Dclose(id);
}

void DataSet::read(span<double> data)
{
	assert(data.size() == size);
	enforce(H5Dread(id, H5T_NATIVE_DOUBLE, 0, 0, 0, data.data()));
}

template <> std::vector<double> DataSet::read<double>()
{
	auto r = std::vector<double>(size);
	read(r);
	return r;
}

void DataSet::write(span<const double> data)
{
	assert(data.size() == size);
	enforce(H5Dwrite(id, H5T_NATIVE_DOUBLE, 0, 0, 0, data.data()));
}

DataFile::~DataFile() { close(); }

DataFile DataFile::create(const std::string &filename)
{
	auto id = enforce(H5Fcreate(filename.c_str(), H5F_ACC_EXCL, 0, 0));
	return DataFile(id);
}

DataFile DataFile::open(const std::string &filename)
{
	auto id = enforce(H5Fopen(filename.c_str(), H5F_ACC_RDWR, 0));
	return DataFile(id);
}

void DataFile::close()
{
	if (id > 0)
		H5Fclose(id);
	id = 0;
}

DataSet DataFile::createData(const std::string &name,
                             const std::vector<hsize_t> &size)
{
	assert(id > 0);
	auto type = H5T_NATIVE_DOUBLE;
	auto space = enforce(H5Screate_simple(size.size(), size.data(), nullptr));
	auto set = enforce(H5Dcreate2(id, name.c_str(), type, space, 0, 0, 0));
	H5Sclose(space);
	return DataSet(set);
}

DataSet DataFile::openData(const std::string &name)
{
	assert(id > 0);
	auto set = enforce(H5Dopen2(id, name.c_str(), 0));
	return DataSet(set);
}

bool DataFile::exists(const std::string &name)
{
	assert(id > 0);
	return enforce(H5Lexists(id, name.c_str(), 0)) > 0;
}

void DataFile::remove(const std::string &name)
{
	assert(id > 0);
	enforce(H5Ldelete(id, name.c_str(), 0));
}

void DataFile::makeGroup(const std::string &name)
{
	assert(id > 0);
	auto group = enforce(H5Gcreate2(id, name.c_str(), 0, 0, 0));
	H5Gclose(group);
}

void DataFile::setAttribute(const std::string &name, hid_t type, const void *v)
{
	assert(id > 0);
	auto space = enforce(H5Screate(H5S_SCALAR));
	auto attr = enforce(H5Acreate2(id, name.c_str(), type, space, 0, 0));
	enforce(H5Awrite(attr, type, v));
	H5Aclose(attr);
	H5Sclose(space);
}

void DataFile::setAttribute(const std::string &name, hid_t type, hsize_t count,
                            const void *v)
{
	assert(id > 0);
	auto space = enforce(H5Screate_simple(1, &count, nullptr));
	auto attr = enforce(H5Acreate2(id, name.c_str(), type, space, 0, 0));
	enforce(H5Awrite(attr, type, v));
	H5Aclose(attr);
	H5Sclose(space);
}

void DataFile::setAttribute(const std::string &name, double v)
{
	setAttribute(name, H5T_NATIVE_DOUBLE, &v);
}

void DataFile::setAttribute(const std::string &name, int v)
{
	setAttribute(name, H5T_NATIVE_INT, &v);
}

void DataFile::setAttribute(const std::string &name, const std::string &v)
{
	auto type = enforce(H5Tcopy(H5T_C_S1));
	enforce(H5Tset_size(type, H5T_VARIABLE));
	const char *ptr = v.c_str();
	setAttribute(name, type, &ptr);
	H5Tclose(type);
}

void DataFile::setAttribute(const std::string &name,
                            const std::vector<double> &v)
{
	setAttribute(name, H5T_NATIVE_DOUBLE, v.size(), v.data());
}

void DataFile::setAttribute(const std::string &name, const std::vector<int> &v)
{
	setAttribute(name, H5T_NATIVE_INT, v.size(), v.data());
}

void DataFile::getAttribute(const std::string &name, hid_t type, void *data)
{
	// open attribute
	auto attr = enforce(H5Aopen(id, name.c_str(), 0));

	// check size
	auto space = enforce(H5Aget_space(attr));
	auto size = H5Sget_simple_extent_npoints(space);
	H5Sclose(space);
	if (size != 1)
		throw std::runtime_error("HDF5 error: wrong attribute size");

	// read attribute
	enforce(H5Aread(attr, type, data));

	H5Aclose(attr);
}

template <> int DataFile::getAttribute<int>(const std::string &name)
{
	int r;
	getAttribute(name, H5T_NATIVE_INT, &r);
	return r;
}

template <> double DataFile::getAttribute<double>(const std::string &name)
{
	double r;
	getAttribute(name, H5T_NATIVE_DOUBLE, &r);
	return r;
}

template <>
std::string DataFile::getAttribute<std::string>(const std::string &name)
{
	auto type = enforce(H5Tcopy(H5T_C_S1));
	enforce(H5Tset_size(type, H5T_VARIABLE));

	char *ptr;
	getAttribute(name, type, &ptr);
	auto r = std::string(ptr);
	free(ptr);

	H5Tclose(type);
	return r;
}

template <>
std::vector<int>
DataFile::getAttribute<std::vector<int>>(const std::string &name)
{
	// open attribute
	auto attr = enforce(H5Aopen(id, name.c_str(), 0));

	// check size
	auto space = enforce(H5Aget_space(attr));
	auto size = H5Sget_simple_extent_npoints(space);
	auto r = std::vector<int>(size);
	H5Sclose(space);

	// read attribute
	enforce(H5Aread(attr, H5T_NATIVE_INT, r.data()));

	H5Aclose(attr);
	return r;
}
