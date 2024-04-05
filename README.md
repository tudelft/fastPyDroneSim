<!--- 
    This is a README.md template for releasing a code project in a GitHub/Gitlab repository.    
    Under each section you can find commented text with explanation on what to add in each section.  
    Please modify the sections depending on needs, and delete all commented text once the README is done.   
    https://github.com/HeatherAn/recommended-coding-practices/blob/main/templates/README_code.md
-->

# fastPyDroneSim

## Description

<!--- Provide description of the contents of the code repository   
    * Provide information about what the code does  
    * Provide links for demos, blog posts, etc. (if applicable)  
    * Mention any caveats and assumptions that were considered  
-->  

Fast drone sim in python (currently linear quadrotors only). Vectorized on CPU
cranking about 5M timesteps per second on a laptop i7. 
Also vectorized for GPU with up to 250M timesteps per second (RTX A1000), but if
the state at every time step needs to be logged and be available to the CPU after simulation, then
that number reduces to 20M or so.

Visuals in THREE.js with a websocket connection, inspired by the Learning to Fly in Seconds paper (https://arxiv.org/abs/2311.13081).

<img src="docs/512quadRotors.gif" width="768">512 quadrotors displayed at the same time</img>

This gif shows the visualization of 512 quadrotors, although 8192 were simulated on a RTX A1000 graphics card.

<!--- Provide a changelog (if applicable)  
## History

-->



## Authors or Maintainers

<!--- Provide information about authors, maintainers and collaborators specifying contact details and role within the project, e.g.:   
    * Full name ([@GitHub username](https://github.com/username), [ORCID](https://doi.org/...), email address, institution/employer (role)  
-->

Till Blaha - [@tblaha](https://github.com/tblaha)



<!--- Provide a table of contents to help readers navigate the README  
## Table of Contents

-->



## Requirements  

Ubuntu 22.04 tested only. Visuals require docker.

Python dependencies can be installed with:

    pip install -r requirements.txt


### CUDA

Cuda can be a hassle to install. For Ubuntu 22.04 on my machine, the following
worked, but there may be (must be!) easier ways, perhaps with a different version.

- Delete all current nvidia things `sudo apt purge "nvidia-*" "libnvidia-*" "cuda-*" "linux-modules-nvidia-*"`
- `sudo apt autoremove`
- Make sure that "Using X.Org X server" is selected in "Software & Updates" > "Additional Drivers"
- Compile the latests 550 driver from scratch for your kernel (`6.1.0-1036-oem` in my case)
```
sudo apt install gcc-12
sudo update-alternatives --install  /usr/bin/cc cc /usr/bin/x86_64-linux-gnu-gcc-12 100
```
- download the runscript from https://www.nvidia.com/download/driverResults.aspx/218826/en-us/. Terminate x-server and run
```
chmod +x ./NVIDIA-Linux-x86_64-550.54.14.run
sudo ./NVIDIA-Linux-x86_64-550.54.14.run        # say yes to everything
```

- reboot and do ONLY the "Base Installer" step of https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
- finally, install cuda and some other things

```
sudo apt install cuda-driver-550 nvidia-compute-utils-550
```


<!--- Add badges of requirements e.g.:  
    [![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)  
-->  

<!--- Provide details of the software required   
    * Add a `requirements.txt` file to the root directory for installing the necessary dependencies.   
    * Describe how to install requirements e.g. when using pip:  

        To install requirements:

        ```
            setup
            pip install -r requirements.txt
        ```

    * Alternatively, create an INSTALL.md. 
    * Provide any further instructions on how others can make sure the scripts are running for benchmarking examples (e.g. by using computational notebooks such as Jupyter notebooks) 
-->


## Running


### Visuals

Build docker image of the webserver for the visuals

    docker build -t devserver . -f dockerfile

Run it with

    docker run -it --net host -v ./static:/usr/app/static devserver

Open the URL reported by the docker container in a WebGL capable browser.


### Simulation

    ipython -i sim.py

Check options at the top of `sim.py` for configuration.



## Structure

<!--- Add here the directory structure of the repo, including file/directory naming conventions  
-->

```
.
├── benchmarks/  # unrelated code for testing numba/cuda
├── libs/        # math functions, compute kernels and visualization interface
├── static/      # the visualization frontend code
├── dockerfile   # sets up a Vite development webserver to host the visuals
├── crafts.py    # contains the drone definitions (currently only quadrotor)
├── sim.py       # simulation main function
├── LICENSE
├── README.md
└── requirements.txt
```

## License

Copyright (C) 2024 Till Blaha -- TU Delft

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


<!--- Add badge for the license under which the code will be released, e.g.:
    [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  
-->

<!--- Place your license text in a file named LICENSE in the root of the repository.  
    * Include information about the license under which the contents of the repository are released publicly. If different licenses apply to different files, explain here which license applies to which file(s), create a LICENSE directory, and add there the licenses (legal text as .md or .txt) for the different files.
    * (If the employer waives its copyright to its employees when the code is released as open-source) Add the copyright statement from the employer of the authors and maintainers (one copyright statement per employer) e.g.: 

    The contents of this repository are licensed under a **Apache License 2.0** license (see LICENSE file).

    Copyright notice:  

    <employer> hereby disclaims all copyright interest in the program “[name_program]” (provide one line description of the content or function) written by the Author(s).  
    <employer representative>, <employer>.  

    © [year_of_release], [name_authors], [reference project, grant or study if desired]  
-->



## References

<!--- Provide links to applicable references    
--> 
Eschmann, Albani and Loianno, "Learning to Fly in Seconds", 2023, [arXiv 2311.13081](https://arxiv.org/abs/2311.13081)


## Citation

<!--- Make the repository citable 
    * If you will be using the Zenodo-Github integration, add the following reference and the DOI of the Zenodo repository:

        If you want to cite this repository in your research paper, please use the following information:
        Reference: [Making Your Code Citable](https://guides.github.com/activities/citable-code/)  

    * If you will be using the 4TU.ResearchData-Github integration, add the following reference and the DOI of the 4TU.ResearchData repository:

        If you want to cite this repository in your research paper, please use the following information:   
        Reference: [Connecting 4TU.ResearchData with Git](https://data.4tu.nl/info/about-your-data/getting-started)   
-->



## Would you like to contribute?

<!--- Add here how you would like others to contribute to this project (e.g. forking, opening issues only, etc.)

    * Do not forget to mention how others can specify how they contributed to the project (e.g., add their names in a separate list of Contributors in the README; add their contributions in separate files specifying their copyright attribution at the top of the source files as commented text; etc.)  
-->
