# D_FAST_all

D_FAST_all is a Python-based project designed to run on the PyCharm platform. This code implements a feature extraction algorithm and similarity search for earthquake signal detection.

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/your_github_name/D_FAST_all">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">D_FAST_all</h3>
  <p align="center">
    A Python project for feature extraction and similarity search in seismic data!
    <br />
    <a href="https://github.com/your_github_name/D_FAST_all"><strong>Explore the documentation »</strong></a>
    <br />
    <br />
    <a href="https://github.com/your_github_name/D_FAST_all">View Demo</a>
    ·
    <a href="https://github.com/your_github_name/D_FAST_all/issues">Report Bug</a>
    ·
    <a href="https://github.com/your_github_name/D_FAST_all/issues">Request Feature</a>
  </p>
</p>

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [File Directory Explanation](#file-directory-explanation)
- [Functionality Overview](#functionality-overview)
- [Deployment](#deployment)
- [Frameworks Used](#frameworks-used)
- [Contributors](#contributors)
  - [How to Contribute](#how-to-contribute)
- [Version Control](#version-control)
- [Author](#author)
- [Acknowledgements](#acknowledgements)

### Getting Started

Please replace all instances of “your_github_name/D_FAST_all” with “your_github_name/your_repository” in the links.

###### Prerequisites

1. Python 3.x
2. PyCharm IDE

###### **Installation Steps**

1. Clone the repository

```sh
git clone https://github.com/GongniNDC/D_FAST_all.git
```

2. Navigate to the directory

```sh
cd D_FAST_all
```

### File Directory Explanation

```
D_FAST_all
├── LEDS/
│   ├── LEDS.py
│   ├── pair_sim_calculate.py
│   └── result_plot.py
├── WMPLSH/
│   ├── index_table.py
│   ├── wmplsh.py
│   └── Roc_plot.py
├── FAST/
│   ├── Fingerprint_Extraction.py
│   ├── LSH.py
│   └── evalution.py
├── preprocess/
├── Releases/
│   └── PNWData1/
├── README.md
└── LICENSE.txt
```

### Functionality Overview

- **LEDS Folder**
  - `LEDS.py`: Used for feature extraction to generate binary fingerprints. The `input_folder` stores waveform data in `.npy` format, and the `output_folder` stores the generated binary fingerprints.
  - `pair_sim_calculate.py`: Calculates pairwise similarity of the fingerprints.
  - `result_plot.py`: Plots the distribution of fingerprint similarities to preliminarily validate the effectiveness of the fingerprint extraction algorithm.

- **WMPLSH Folder**
  - `index_table.py`: Generates a hash index table.
  - `wmplsh.py`: Performs similarity search and statistics of detection results.
  - `Roc_plot.py`: Plots the results of the detection.

- **FAST Folder**
  - `Fingerprint_Extraction.py`: Implements the original FAST algorithm for extracting sparse binary fingerprints.
  - `LSH.py`: Conducts similarity search using the original FAST algorithm.
  - `evalution.py`: Compares the performance of the D_FAST and original FAST algorithms and plots the results.

### Deployment

No deployment instructions are currently provided.

### Frameworks Used
This project relies on the following frameworks and libraries:

NumPy: A fundamental package for scientific computing with Python.
Pandas: A data analysis and manipulation library.
Matplotlib: A plotting library for creating static, animated, and interactive visualizations in Python.
ObsPy: A Python toolbox for seismology.

### Contributors

Please refer to **CONTRIBUTING.md** for a list of developers who contributed to this project.

#### How to Contribute

Contributions make the open-source community a great place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Version Control

This project uses Git for version control. You can check the available versions in the repository.

### Author

Gongni:gong.ni@ndc.org.cn


*You can also refer to the contributors list for all developers involved in this project.*

### License

This project is licensed under the MIT License. For details, please refer to [LICENSE.txt](https://github.com/your_github_name/D_FAST_all/blob/master/LICENSE.txt).

### Acknowledgements

- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)


Please replace placeholders with actual information relevant to your project.