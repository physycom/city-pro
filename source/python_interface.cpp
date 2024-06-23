#include <pybind11/pybind11.h>
#include <main_city-pro.h>
namespace py = pybind11;
// Module to Be Called by Python import statement 
// Arg[0] -> Module Name
// Arg[1] -> Module Object from py::module_
PYBIND11_MODULE(city_pro, m)
{
    m.def("main_city_pro", &main_city_pro,"Function that launches the analysis of data for a given date",
    py::arg("argc"),py::arg("argv"))
}

