#import <Python.h>

static char module_desc[] = "This is an SOM Implementation";

static PyObject * 
_som_create(PyObject *self, PyObject *args)
{
	int width;
	int height;

   if (!PyArg_ParseTuple(args, "ii", &width, &height))
       return NULL;
	printf("[%i,%i]",width,height);
	Py_INCREF(Py_None);
   return Py_None;
}


static PyMethodDef module_methods[] = {
    {"create", _som_create, METH_VARARGS, "Create an internal _som object"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC init_som(void)
{
	PyObject *m =  Py_InitModule3("_som",module_methods,module_desc);
	if (m == NULL)
		return;
}

