#ifndef ENV_HEADER
#define ENV_HEADER
#include <string>
#include <variant>
#include <vector>
#include <map>
#include "Python.h"

class Env {
  public:
    PyObject* envModule;
    PyObject* env;

    Env(){};
    ~Env(){};

    void init(
        std::string game,
        std::map<std::string, std::variant<double, std::string, std::vector<int>, int>> params);
    std::tuple<std::vector<double>, double, bool> step(int action);
    std::vector<double> reset();
    void cleanup();
    std::unique_ptr<Env> clone();
};

void py_finalize();
#endif
