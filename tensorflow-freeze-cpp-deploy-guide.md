# TF/C++ Deployment

Guide to freeze graph and deploy in TF C++
[link](https://medium.com/@hamedmp/exporting-trained-tensorflow-models-to-c-the-right-way-cf24b609d183)


The guide follows the same process as [tensorflow-freeze-guide.md](tensorflow-freeze-guide.md). 
Additional step is to modify the `label_image` tf/c++ example for custom application, and `bazel build` it.
