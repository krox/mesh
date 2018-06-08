
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

DataSet::~DataSet()
{
	if (id > 0)
		H5Dclose(id);
}

void DataSet::read(span<double> data)
{
	enforce(H5Dread(id, H5T_NATIVE_DOUBLE, 0, 0, 0, data.data()));
}

void DataSet::write(span<const double> data)
{
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
