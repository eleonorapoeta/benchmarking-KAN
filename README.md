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

Note: running the experiments will currently reproduce the results presented in the paper, i.e. run the selected model(s) (mlp, kan, or both) for all model sizes considered, for a total of 5 times each (to quantify the variance of the model).

Some information on the machine used:
- CPU: Intel(R) Core(TM) i9-10980XE
- RAM: 128 GB DDR4 (4x32GB)
- GPUs: NVIDIA A6000 x2
- OS: Ubuntu 22.04
- Python version: 3.10.12


# ❌ Troubleshoting
Some errors that may occur are listed below:

### PAPI\_EPERM
We use Performance Application Programming Interface (PAPI) to evaluate the performance of the models in terms of instructions executed.

The following error may occur when running the code. 
```pypapi.exceptions.PapiPermissionError: Permission level does not permit operation. (PAPI_EPERM)```

Fix: 
1. Run the code as superuser (root)
2. Adjust the kernel parameters to allow non-root access to performance counters. You can temporarily set these parameters:
```bash
sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
```
3. For a more permanent solution, add the following lines to `/etc/sysctl.conf`:
```bash
kernel.perf_event_paranoid = -1
kernel.perf_event_max_sample_rate = 100000
```
Then apply the changes:
```bash
sudo sysctl -p
```

# 📜  License
This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.


# 📝 TODO
- [ ] Allow custom model sizes through arguments
- [ ] Allow custom number of runs (seeds) through arguments
- [ ] Make the computation of the performance (via PAPI) optional (it may cause some errors if not enough permissions are available)
- [ ] Add a [Contributing Guide](./CONTRIBUTING.md) 🙃
- [ ] Describe the output format in the documentation
- [ ] Introduce a better format for the output files

# 🤝 Contributing
Contributions, issues, and feature requests are welcome! See our [Contributing Guide](./CONTRIBUTING.md) for more details.
