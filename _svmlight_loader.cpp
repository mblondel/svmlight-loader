/*
 * Authors: Mathieu Blondel <mathieu@mblondel.org>
 *          Lars Buitinck <L.J.Buitinck@uva.nl>
 *
 * License: Simple BSD
 *
 * This module implements _load_svmlight_format, a fast and memory efficient
 * function to load the file format originally created for svmlight and now used
 * by many other libraries, including libsvm.
 *
 * The function loads the file directly in a CSR sparse matrix without memory
 * copying.  The approach taken is to use 4 C++ vectors (data, indices, indptr
 * and labels) and to incrementally feed them with elements. Ndarrays are then
 * instantiated by PyArray_SimpleNewFromData, i.e., no memory is
 * copied.
 *
 * Since the memory is not allocated by the ndarray, the ndarray doesn't own the
 * memory and thus cannot deallocate it. To automatically deallocate memory, the
 * technique described at http://blog.enthought.com/?p=62 is used. The main idea
 * is to use an additional object that the ndarray does own and that will be
 * responsible for deallocating the memory.
 */


#include <Python.h>
#include <numpy/arrayobject.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


/*
 * A Python object responsible for memory management of our vectors.
 */
template <typename T>
struct VectorOwner {
  // Inherit from the base Python object.
  PyObject_HEAD
  // The vector that VectorOwner is responsible for deallocating.
  std::vector<T> v;
};

/*
 * Deallocator template.
 */
template <typename T>
static void destroy_vector_owner(PyObject *self)
{
  // Note: explicit call to destructor because of placement new in
  // to_1d_array. memory management for VectorOwner is performed by Python.
  // Compiler-generated destructor will release memory from vector member.
  VectorOwner<T> &obj = *reinterpret_cast<VectorOwner<T> *>(self);
  obj.~VectorOwner<T>();

  self->ob_type->tp_free(self);
}

/*
 * Since a template function can't have C linkage,
 * we instantiate the template for the types "int" and "double"
 * in the following two functions. These are used for the tp_dealloc
 * attribute of the vector owner types further below.
 */
extern "C" {
static void destroy_int_vector(PyObject *self)
{
  destroy_vector_owner<int>(self);
}

static void destroy_double_vector(PyObject *self)
{
  destroy_vector_owner<double>(self);
}
}


/*
 * Type objects for above.
 */
static PyTypeObject IntVOwnerType    = { PyObject_HEAD_INIT(NULL) },
                    DoubleVOwnerType = { PyObject_HEAD_INIT(NULL) };

/*
 * Set the fields of the owner type objects.
 */
static void init_type_objs()
{
  IntVOwnerType.tp_flags = DoubleVOwnerType.tp_flags = Py_TPFLAGS_DEFAULT;
  IntVOwnerType.tp_name  = DoubleVOwnerType.tp_name  = "deallocator";
  IntVOwnerType.tp_doc   = DoubleVOwnerType.tp_doc   = "deallocator object";
  IntVOwnerType.tp_new   = DoubleVOwnerType.tp_new   = PyType_GenericNew;

  IntVOwnerType.tp_basicsize     = sizeof(VectorOwner<int>);
  DoubleVOwnerType.tp_basicsize  = sizeof(VectorOwner<double>);
  IntVOwnerType.tp_dealloc       = destroy_int_vector;
  DoubleVOwnerType.tp_dealloc    = destroy_double_vector;
}

PyTypeObject &vector_owner_type(int typenum)
{
  switch (typenum) {
    case NPY_INT: return IntVOwnerType;
    case NPY_DOUBLE: return DoubleVOwnerType;
  }
  throw std::logic_error("invalid argument to vector_owner_type");
}


/*
 * Convert a C++ vector to a 1d-ndarray WITHOUT memory copying.
 * Steals v's contents, leaving it empty.
 * Throws an exception if an error occurs.
 */
template <typename T>
static PyObject *to_1d_array(std::vector<T> &v, int typenum)
{
  npy_intp dims[1] = {v.size()};

  // A C++ vector's elements are guaranteed to be in a contiguous array.
  PyObject *arr = PyArray_SimpleNewFromData(1, dims, typenum, &v[0]);

  try {
    if (!arr)
      throw std::bad_alloc();

    VectorOwner<T> *owner = PyObject_New(VectorOwner<T>,
                                         &vector_owner_type(typenum));
    if (!owner)
      throw std::bad_alloc();

    // Transfer ownership of v's contents to the VectorOwner.
    // Note: placement new.
    new (&owner->v) std::vector<T>();
    owner->v.swap(v);

    PyArray_BASE(arr) = (PyObject *)owner;

    return arr;

  } catch (std::exception const &e) {
    // Let's assume the Python exception is already set correctly.
    Py_XDECREF(arr);
    throw;
  }
}


static PyObject *to_csr(std::vector<double> &data,
                        std::vector<int> &indices,
                        std::vector<int> &indptr,
                        std::vector<double> &labels)
{
  // We could do with a smart pointer to Python objects here.
  std::exception const *exc = 0;
  PyObject *data_arr = 0,
           *indices_arr = 0,
           *indptr_arr = 0,
           *labels_arr = 0,
           *ret_tuple = 0;

  try {
    data_arr    = to_1d_array(data, NPY_DOUBLE);
    indices_arr = to_1d_array(indices, NPY_INT);
    indptr_arr  = to_1d_array(indptr, NPY_INT);
    labels_arr  = to_1d_array(labels, NPY_DOUBLE);

    ret_tuple = Py_BuildValue("OOOO",
                              data_arr, indices_arr,
                              indptr_arr, labels_arr);
  } catch (std::exception const &e) {
    exc = &e;
  }

  // Py_BuildValue increases the reference count of each array,
  // so we need to decrease it before returning the tuple,
  // regardless of error status.
  Py_XDECREF(data_arr);
  Py_XDECREF(indices_arr);
  Py_XDECREF(indptr_arr);
  Py_XDECREF(labels_arr);

  if (exc)
    throw *exc;

  return ret_tuple;
}


/*
 * Parsing.
 */

class SyntaxError : public std::runtime_error {
public:
  SyntaxError(std::string const &msg)
   : std::runtime_error(msg + " in SVMlight/libSVM file")
  {
  }
};

/*
 * Parse single line. Throws exception on failure.
 */
void parse_line(const std::string& line,
                std::vector<double> &data,
                std::vector<int> &indices,
                std::vector<int> &indptr,
                std::vector<double> &labels)
{
  if (line.length() == 0)
    throw SyntaxError("empty line");

  if (line[0] == '#')
    return;

  // FIXME: we shouldn't be parsing line-by-line.
  // Also, we might catch more syntax errors with failbit.
  std::istringstream in(line);
  in.exceptions(std::ios::badbit);

  double y;
  if (!(in >> y)) {
    throw SyntaxError("non-numeric or missing label");
  }

  labels.push_back(y);
  indptr.push_back(data.size());

  char c;
  double x;
  unsigned idx;

  while (in >> idx >> c >> x) {
    if (c != ':')
      throw SyntaxError(std::string("expected ':', got '") + c + "'");
    indices.push_back(int(idx));
    data.push_back(x);
  }
}

/*
 * Parse entire file. Throws exception on failure.
 */
void parse_file(char const *file_path,
                size_t buffer_size,
                std::vector<double> &data,
                std::vector<int> &indices,
                std::vector<int> &indptr,
                std::vector<double> &labels)
{
  std::vector<char> buffer(buffer_size);

  std::ifstream file_stream;
  file_stream.exceptions(std::ios::badbit);
  file_stream.rdbuf()->pubsetbuf(&buffer[0], buffer_size);
  file_stream.open(file_path);

  if (!file_stream)
    throw std::ios_base::failure("File doesn't exist!");

  std::string line;
  while (std::getline(file_stream, line))
    parse_line(line, data, indices, indptr, labels);
  indptr.push_back(data.size());
}

/*
 * Parse entire string. Throws exception on failure.
 */
void parse_string(char const *s,
                  std::vector<double> &data,
                  std::vector<int> &indices,
                  std::vector<int> &indptr,
                  std::vector<double> &labels)
{
  std::istringstream string_stream;
  string_stream.rdbuf()->pubsetbuf(const_cast<char *>(s), strlen(s));

  std::string line;
  while (std::getline(string_stream, line))
    parse_line(line, data, indices, indptr, labels);
  indptr.push_back(data.size());
}

static const char load_svmlight_file_doc[] =
  "Load file in svmlight format and return a CSR.";

extern "C" {
static PyObject *load_svmlight_file(PyObject *self, PyObject *args)
{
  try {
    // Read function arguments.
    char const *file_path;
    int buffer_mb;

    if (!PyArg_ParseTuple(args, "si", &file_path, &buffer_mb))
      return 0;

    buffer_mb = std::max(buffer_mb, 1);
    size_t buffer_size = buffer_mb * 1024 * 1024;

    std::vector<double> data, labels;
    std::vector<int> indices, indptr;
    parse_file(file_path, buffer_size, data, indices, indptr, labels);

    return to_csr(data, indices, indptr, labels);

  } catch (SyntaxError const &e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return 0;
  } catch (std::bad_alloc const &e) {
    PyErr_SetString(PyExc_MemoryError, e.what());
    return 0;
  } catch (std::ios_base::failure const &e) {
    PyErr_SetString(PyExc_IOError, e.what());
    return 0;
  } catch (std::exception const &e) {
    std::string msg("error in SVMlight/libSVM reader: ");
    msg += e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());
    return 0;
  }
}
}

static const char load_svmlight_string_doc[] =
  "Parse string in svmlight format and return a CSR.";

extern "C" {
static PyObject *load_svmlight_string(PyObject *self, PyObject *args)
{
  try {
    // Read function arguments.
    char const *s;

    if (!PyArg_ParseTuple(args, "s", &s))
      return 0;

    std::vector<double> data, labels;
    std::vector<int> indices, indptr;
    parse_string(s, data, indices, indptr, labels);

    return to_csr(data, indices, indptr, labels);

  } catch (SyntaxError const &e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return 0;
  } catch (std::bad_alloc const &e) {
    PyErr_SetString(PyExc_MemoryError, e.what());
    return 0;
  } catch (std::ios_base::failure const &e) {
    PyErr_SetString(PyExc_IOError, e.what());
    return 0;
  } catch (std::exception const &e) {
    std::string msg("error in SVMlight/libSVM reader: ");
    msg += e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());
    return 0;
  }
}
}

static const char dump_svmlight_file_doc[] =
  "Dump CSR matrix to a file in svmlight format.";

extern "C" {
static PyObject *dump_svmlight_file(PyObject *self, PyObject *args)
{
  try {
    // Read function arguments.
    char const *file_path;
    PyArrayObject *indices_array, *indptr_array, *data_array, *label_array;
    int zero_based;

    if (!PyArg_ParseTuple(args,
                          "sO!O!O!O!i",
                          &file_path,
                          &PyArray_Type, &data_array,
                          &PyArray_Type, &indices_array,
                          &PyArray_Type, &indptr_array,
                          &PyArray_Type, &label_array,
                          &zero_based))
      return 0;

    int n_samples = indptr_array->dimensions[0] - 1;
    double *data = (double*) data_array->data;
    int *indices = (int*) indices_array->data;
    int *indptr = (int*) indptr_array->data;
    double *y = (double*) label_array->data;

    std::ofstream fout;
    fout.open(file_path, std::ofstream::out);

    int idx;
    for (int i=0; i < n_samples; i++) {
      fout << y[i] << " ";
      for (int jj=indptr[i]; jj < indptr[i+1]; jj++) {
        idx = indices[jj];
        if (!zero_based)
          idx++;
        fout << idx << ":" << data[jj] << " ";
      }
      fout << std::endl;
    }

    fout.close();

    Py_INCREF(Py_None);
    return Py_None;

  } catch (std::exception const &e) {
    std::string msg("error in SVMlight/libSVM writer: ");
    msg += e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());
    return 0;
  }
}
}


/*
 * Python module setup.
 */

static PyMethodDef svmlight_format_methods[] = {
  {"_load_svmlight_file", load_svmlight_file,
    METH_VARARGS, load_svmlight_file_doc},

  {"_load_svmlight_string", load_svmlight_string,
    METH_VARARGS, load_svmlight_string_doc},

  {"_dump_svmlight_file", dump_svmlight_file,
    METH_VARARGS, dump_svmlight_file_doc},

  {NULL, NULL, 0, NULL}
};

static const char svmlight_format_doc[] =
  "Loader/Writer for svmlight / libsvm datasets - C++ helper routines";

extern "C" {
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef _svmlight_loader_definition = { 
    PyModuleDef_HEAD_INIT,
    "_svmlight_loader",
    svmlight_format_doc,
    -1, 
    svmlight_format_methods
};

PyMODINIT_FUNC PyInit__svmlight_loader(void)
{
  Py_Initialize();
  import_array();

  init_type_objs();
  if (PyType_Ready(&DoubleVOwnerType) < 0
  || PyType_Ready(&IntVOwnerType)    < 0)
  return NULL;

  return PyModule_Create(&_svmlight_loader_definition);

}
#else 
PyMODINIT_FUNC init_svmlight_loader(void)
{
  _import_array();

  init_type_objs();
  if (PyType_Ready(&DoubleVOwnerType) < 0
   || PyType_Ready(&IntVOwnerType)    < 0)
    return;

  Py_InitModule3("_svmlight_loader",
                 svmlight_format_methods,
                 svmlight_format_doc);
}
#endif
}