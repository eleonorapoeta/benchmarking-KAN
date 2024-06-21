# Benchmarking-KAN
 This repository contains the official implementation of "[A Benchmarking Study of Kolmogorov-Arnold Networks on Tabular Data](https://arxiv.org/pdf/2406.14529)" (under review). You can use this codebase to replicate our experiments about benchmarking KAN networks on some of the most used real-world tabular datasets.

# 👀 Overview
Kolmogorov-Arnold Networks (KAN) has recently been introduced and gained much attention. In this work, we propose a benchmarking of KAN over some of the most used real-world datasets from [UCI Machine Learning repository](https://archive.ics.uci.edu). We used the implementation of efficient KAN available [here](https://github.com/Blealtan/efficient-kan).


# ⚡️ Quick start
1. Clone this repository: `git clone https://github.com/eleonorapoeta/benchmarking-KAN.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`

# 💻 Reproduce the Experiments
To reproduce the experiments conducted in our study, follow these steps:

After following the Quick Start guide, you can run `python main.py` specifying the following arguments:

- `--model_name` = (kan, mlp, all) depending on the model you want to test.
- `--dataset_name` = one of the tested datasets from UCI or yours.
- `--num_epochs` = epochs of training.


# 📝 License
This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.

# 🤝 Contributing
Contributions, issues, and feature requests are welcome! See our [Contributing Guide](./CONTRIBUTING.md) for more details.
