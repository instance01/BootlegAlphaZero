#include <tuple>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include "Python.h"

#include "env.hpp"
#include "params.hpp"


void
Env::init(
    std::string game,
    Params params
) {
  Py_Initialize();

  PyObject *sys_path = PySys_GetObject("path");
  PyList_Append(sys_path, PyUnicode_FromString(std::filesystem::current_path().c_str()));

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
}


std::tuple<std::vector<double>, double, bool>
Env::step(int action) {
  PyObject* stepFunc = PyObject_GetAttrString(envModule, "step");
  PyObject* result = PyObject_CallFunction(stepFunc, "Oi", env, action);

  if (result == NULL)
    PyErr_Print();

  auto py_obs = PyTuple_GetItem(result, 0);
  auto py_reward = PyTuple_GetItem(result, 1);
  auto py_done = PyTuple_GetItem(result, 2);

  std::vector<double> obs;
  double reward = PyFloat_AsDouble(py_reward);
  bool done = PyLong_AsLong(py_done);

  for(Py_ssize_t i = 0; i < PyList_Size(py_obs); ++i) {
    PyObject *value = PyList_GetItem(py_obs, i);
    obs.push_back(PyFloat_AsDouble(value));
  }

  return std::make_tuple(obs, reward, done);
}

std::vector<double>
Env::reset() {
  std::vector<double> obs;

  PyObject* resetFunc = PyObject_GetAttrString(envModule, "reset");
  PyObject* py_obs = PyObject_CallFunction(resetFunc, "O", env);

  if (py_obs == NULL)
    PyErr_Print();

  for(Py_ssize_t i = 0; i < PyList_Size(py_obs); ++i) {
    PyObject *value = PyList_GetItem(py_obs, i);
    obs.push_back(PyFloat_AsDouble(value));
  }

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
  Env ret = Env();
  ret.envModule = envModule;
  PyObject* cloneFunc = PyObject_GetAttrString(envModule, "clone");
  ret.env = PyObject_CallFunction(cloneFunc, "O", env);
  if (env == NULL)
    PyErr_Print();
  return std::make_unique<Env>(ret);
}
