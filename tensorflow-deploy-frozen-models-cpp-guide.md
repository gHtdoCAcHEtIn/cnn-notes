# TF/C++ Deployment

Guide to freeze graph and deploy in TF C++
[link](https://medium.com/@hamedmp/exporting-trained-tensorflow-models-to-c-the-right-way-cf24b609d183)


## Freeze

* Follow the guide in [tensorflow-slim-freeze-models-guide.md](tensorflow-slim-freeze-models-guide.md) for freezing a tensorflow model.

## Deploy

* For deploy, duplicate the `label_image` tf/c++ example, and modify for custom application.
* Build your application using `bazel build`.
