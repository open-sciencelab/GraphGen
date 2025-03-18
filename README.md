# GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation

[English](README.md) | [简体中文](README_zh.md)

<p align="center">
  <img src="resources/images/favicon.png" width="200" height="200" style="border-radius: 50%">
</p>

```text
├── baselines/           # baseline methods
├── cache/               # cache files
│   ├── data/            # generated data
│   ├── logs/            # log files
├── configs/             # configuration files
├── graphgen/            # GraphGen implementation
│   ├── operators/       # operators
│   ├── graphgen.py      # main file
├── models/              # base classes
├── resources/           # static files and examples
├── scripts/             # scripts for running experiments
├── templates/           # prompt templates
├── utils/               # utility functions
├── webui/               # web interface
└── README.md
```

## Introduction

GraphGen is a framework for synthetic data generation guided by knowledge graphs.

### Workflow
![workflow](resources/images/flow.png)

### User Interface
![ui](resources/images/interface.jpg)
