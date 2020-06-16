#include <tuple>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <chrono>
#include "Python.h"

#include <unistd.h>
#include <limits.h>

#include "env.hpp"
#include "params.hpp"


void
Env::init(
    std::string game,
    Params params
) {
  Py_Initialize();

  PyObject *sys_path = PySys_GetObject("path");
  char cwd[PATH_MAX];
  getcwd(cwd, sizeof(cwd));
  const char* path = &cwd[0];
  PyList_Append(sys_path, PyUnicode_FromString(path));

  PyObject* py_game = PyUnicode_FromString(game.c_str());
  PyObject* py_params = PyDict_New();

  for (auto it = params.begin(); it != params.end(); it++) {
    auto key = it->first;
    auto value = it->second;

    PyObject* py_val;

    if (std::holds_alternative<int>(value)) {
      auto v = *std::get_if<int>(&value);
      py_val = PyLong_FromLong(v);
    } else if (std::holds_alternative<double>(value)) {
      auto v = *std::get_if<double>(&value);
      py_val = PyFloat_FromDouble(v);
    } else if (std::holds_alternative<std::vector<int>>(value)) {
      auto v = *std::get_if<std::vector<int>>(&value);
      //py_val = PyList_New(v.size());
      // What a dumb mistake that was. This resulted in a list full of NULLs.
      // And I kept appending to that.
      // So when the deepcopy (see clone()) tried to copy my faulty env object with params
      // containing NULLs, of course it segfaulted.
      py_val = PyList_New(0);
      for (int i : v) {
        PyObject* py_int = PyLong_FromLong(i);
        PyList_Append(py_val, py_int);
      }
    } else if (std::holds_alternative<std::string>(value)) {
      auto v = *std::get_if<std::string>(&value);
      py_val = PyUnicode_FromString(v.c_str());
    } else {
      continue;
    }

    PyDict_SetItemString(py_params, key.c_str(), py_val);
  }

  envModule = PyImport_ImportModule("env");
  if (envModule == NULL)
    PyErr_Print();

  PyObject* initFunc = PyObject_GetAttrString(envModule, "init");
  env = PyObject_CallFunction(initFunc, "OO", py_game, py_params);
  if (env == NULL)
    PyErr_Print();

  stepFunc = PyObject_GetAttrString(envModule, "step");
  resetFunc = PyObject_GetAttrString(envModule, "reset");
  cloneFunc = PyObject_GetAttrString(envModule, "clone");
}

std::tuple<std::vector<double>, double, bool>
Env::step(int action) {
  std::vector<double> obs;
  double reward;
  bool done;

  //auto t1 = std::chrono::high_resolution_clock::now();
  auto py_action = PyLong_FromLong(action);

  //PyObject* result = PyObject_CallFunction(stepFunc, "Oi", env, action);
  //PyObject* result = PyObject_CallFunctionObjArgs(stepFunc, env, py_action, NULL);

  std::vector<PyObject*> o({env, py_action});

  //auto f = _PyCCall_FastCall(stepFunc, o.data(), 2, NULL);

  // Requires Python 3.8.
  PyObject* result = _PyObject_Vectorcall(stepFunc, o.data(), 2, NULL);

  //auto args = PyTuple_New(2);
  //Py_INCREF(env);
  //PyTuple_SET_ITEM(args, 0, env);
  //Py_INCREF(py_action);
  //PyTuple_SET_ITEM(args, 1, py_action);
  //Py_INCREF(stepFunc);
  ////result = __Pyx_PyObject_Call(stepFunc, args, NULL);
  //ternaryfunc call = stepFunc->ob_type->tp_call;
  //auto result = (*call)(stepFunc, args, NULL);

  Py_DECREF(py_action);

  // Calling Python functions was measured here. In a Docker with 3 GB RAM and 2 cores on a Mac it
  // takes 500 microseconds to call a Python function.
  // For reference, it takes 50 microseconds to do the step in Python.
  // The rest of the application cannot bring this performance penalty back. In the end, with a
  // Python bridge the C++ version is up to 6 times slower than the Python version.
  // This makes sense since the Python bridge is used heavily.
  // The clal above was tested without multithreading. This brings more challenges.
  //
  //auto t2 = std::chrono::high_resolution_clock::now();
  //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  //std::cout << "" << duration << std::endl;

  if (result == NULL)
    PyErr_Print();

  auto py_obs = PyTuple_GET_ITEM(result, 0);
  auto py_reward = PyTuple_GET_ITEM(result, 1);
  auto py_done = PyTuple_GET_ITEM(result, 2);

  reward = PyFloat_AsDouble(py_reward);
  done = PyLong_AsLong(py_done);

  for(Py_ssize_t i = 0; i < PyList_Size(py_obs); ++i) {
    PyObject *value = PyList_GetItem(py_obs, i);
    obs.push_back(PyFloat_AsDouble(value));
  }


  Py_DECREF(result);


  return std::make_tuple(obs, reward, done);
}

std::vector<double>
Env::reset() {
  std::vector<double> obs;

  PyObject* py_obs = PyObject_CallFunction(resetFunc, "O", env);

  if (py_obs == NULL)
    PyErr_Print();

  for(Py_ssize_t i = 0; i < PyList_Size(py_obs); ++i) {
    PyObject *value = PyList_GetItem(py_obs, i);
    obs.push_back(PyFloat_AsDouble(value));
  }

  Py_DECREF(py_obs);

  return obs;
}

void
Env::cleanup() {
  // TODO Check that no decrefs are missing !!
  Py_DECREF(envModule);
  Py_DECREF(env);
}

void
py_finalize() {
  Py_FinalizeEx();
}

std::unique_ptr<Env>
Env::clone() {
  Env ret;

  ret = Env();
  ret.envModule = envModule;
  ret.resetFunc = resetFunc;
  ret.cloneFunc = cloneFunc;
  ret.stepFunc = stepFunc;
  ret.env = PyObject_CallFunction(cloneFunc, "O", env);
  if (env == NULL)
    PyErr_Print();

  return std::make_unique<Env>(ret);
}
