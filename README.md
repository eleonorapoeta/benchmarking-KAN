# Benchmarking-KAN
 This repository contains the officil implementation of "Real-World Performance of Kolmogorov-Arnold Networks: A Benchmarking Study" (under review). You can use this codebase to replicate our experiments about benchmarking KAN networks on some of the molst used real-world tabular datasets.

# 👀 Overview
Kolmogorov-Arnold Networks (KAN) have been recently introduced and gained a lot of attention. In this work we propose a benchmarking of KAN over some of the most used real-world datasets from [UCI Machine Learning repository](https://archive.ics.uci.edu)


# ⚡️ Quick start
1. Clone this repository: `git clone https://github.com/eleonorapoeta/benchmarking-KAN.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`

# 💻 Reproduce the Experiments
To reproduce the experiments conducted in our study, follow these steps:

After following the Quick Start guide, you can run `python main.py` specifying the following arguments:

- `--model_name` = (kan, mlp, all) depending what the model you want to test.
- `--dataset_name` = one of the tested dataset from UCI or your datasets.
- `--num_epochs` = epochs of training.


# 📝 License
This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.

# 🤝 Contributing
Contributions, issues, and feature requests are welcome! See our [Contributing Guide](./CONTRIBUTING.md) for more details.
