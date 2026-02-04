---
title: Ocean Color Environment on Server
tags:
  - Ocean Color
  - Linux
categories: Learning Notes
abbrlink: eff06e8d
date: 2021-12-06 15:16:53
mathjax:
copyright:
password:
lang: en
---

Python, SeaDAS&OCSSW, SNAP

<!-- more -->

# Python

At the beginning, I accidentally installed some packages in base, which can be deleted. Try to run in pysoc environment.

```bash
$ mkdir -p miniconda
$ cd miniconda
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda.... .sh
$ conda create --name pysoc python=3.7
$ conda activate pysoc
$ conda install -c conda-forge py6s
$ conda install seaborn
$ conda install -c conda-forge xarray dask netCDF4 bottleneck
$ conda install -c conda-forge cartopy
$ conda install -c conda-forge pyresample
$ conda install -c anaconda basemap
$ conda install -c conda-forge basemap-data-hires
$conda install opencv-python
```

After installation, it's necessary to test py6s

```python
$ python
>>> from Py6S import *
>>> SixS.test()
6S wrapper script by Robin Wilson
Using 6S located at <PATH_TO_SIXS_EXE>
Running 6S using a set of test parameters
The results are:
Expected result: 619.158000
Actual result: 619.158000
#### Results agree, Py6S is working correctly
```

If it shows this, it's correct.

## Jupyter

For convenience in writing programs, I installed jupyter

```bash
$ pip install jupyterlab
$ conda install -c conda-forge nodejs
$ conda install -c conda-forge python-lsp-server r-languageserver
$ jupyter labextension install @jupyterlab/toc
$ conda install -c conda-forge jupyterlab-drawio
$ pip install jupyterlab_latex
$ pip install lckr-jupyterlab-variableinspector
$ conda install -c conda-forge jupyterlab_execute_time
$ conda install -c conda-forge nb_conda_kernels
$ jupyter lab build
$ jupyter labextension enable all
```

Since this is on a remote server, some configuration is needed

```bash
$ jupyter lab --port=8888 --no-browser --allow-root
```

On your own computer

```bash
$ ssh -N -f -L 8880:localhost:8888 root@123435
```

Finally, just add the URL shown on the server in your browser.

Remember to ask the programmer to open the firewall.

# SeaDAS

## Basic Environment

```bash
$ sudo yum -y install gcc
$ sudo yum -y install git
$ sudo yum -y install curl
$ sudo yum -y install wget
```

Since it's RedHat, there's no apt-get.

```bash
$ sudo yum -y-repository ppa:webupd8team/java
$ sudo yum -y update
$ sudo yum -y install oracle-java8-installer
$ sudo yum -y install oracle-java8-set-default
$ java -version
```

Shows

```bash
java version "1.8.0_311"
Java(TM) SE Runtime Environment (build 1.8.0_311-b11)
Java HotSpot(TM) 64-Bit Server VM (build 25.311-b11, mixed mode)
```

```bash
$ wget https://oceandata.sci.gsfc.nasa.gov/SeaDAS/installer/7.3.1/seadas_7.3_linux64_installer.sh

$ bash seadas...sh

```

This is the last version that doesn't require GUI installation, no wonder so many papers keep using 7.3.

Where should SeaDAS be installed?
[/opt/seadas-7.3]

>A strange thing
>
>This is the first time I discovered that Linux paths work this way
>
>cd / enters the root directory
>
>This directory is shared by all users
>
>While cd ~ enters my own directory
>
>Learned something new

## OCSSW

The most important part

```bash
$　wget https://oceandata.sci.gsfc.nasa.gov/ocssw/install_ocssw.py
$ chmod +x install_ocssw.py
$ ./install_ocssw.py --install-dir=/root/SeaDAS/seadas-7.3/ocssw --aqua --seawifs --goci
$ mv /opt/seadas-7.3 ~/SeaDAS
$ vim ~ .bashrc
```

Enter

```bash
export SEADASPATH=/root/SeaDAS/seadas-7.3
export PATH=$PATH:$SEADASPATH/bin
```

where: `[SeaDAS_install_dir]` is the directory where you installed SeaDAS.

Save after finishing.

At this point, I suddenly found that I couldn't find those installed files in the OCSSW path. After searching, it turned out they were installed to / instead of ~. The download location above can change the path but I forgot to change it.

Had to move them over again.

Test command

```bash
$ source .../OCSSW_bash.env
$ which l2gen
```

Shows

```bash
/root/SeaDAS/seadas-7.3/ocssw/bin/l2gen
```

That means it's done.

Then I added a more complete configuration in bashrc

```bash
export SEADASPATH=~/SeaDAS/seadas-7.3
export PATH=$PATH:$SEADASPATH/bin
export PYTHON_PROGRAMS=/root/miniconda3/bin/python
export PYTHONPATH=$PYTHONPATH:$PYTHON_PROGRAMS:$PYTHON_PROGRAMS/utilities
export LOCAL_RESOURCES=$PYTHON_PROGRAMS/local_procesing_resources
export OCSSWROOT=$SEADASPATH/ocssw
export OCSSW_DEBUG=0 # set to 1 for debugging
source $OCSSWROOT/OCSSW_bash.env
```

## File Transfer

Previously I used Putty or FileZilla directly, but the company programmer said to use SCP

```bash
$ scp SourceFile user@server:directory/TargetFile
```

To transfer back

```bash
$ scp user@server:directory/SourceFile TargetFile
```

## Atmospheric Correction

Due to NASA data format updates, it's slightly different from the previous script

```bash
#Example script to process L1A files up to L2, put in same directory as
# your L1A files
# Run script by typing on Terminal Command Line:
# bash l1a_to_l2.sh input.txt
# input.txt contains list of files [ create by: ls -1 *L1A_LAC* > input.txt ]

LIST=$1
if [ -z "$LIST" ]
  then
    echo "No input file list supplied"
    exit 1
fi

if [[ ! -a $LIST ]]; then
    echo "$LIST does not exist!"
    exit 1
fi

mkdir -p done
mkdir -p l2_lac

for FILE in $(cat $LIST);
do
    echo "Working on $FILE"
    echo ${FILE}
    # get file basename (no file extension)
    BASE=`basename $FILE .L1A_LAC`
    echo ${BASE}
    GEOFILE=${BASE}.GEO
    echo ${GEOFILE}
    L1BFILE=${BASE}.L1B_LAC
    echo ${L1BFILE}
    L2FILE=${BASE}.L2_LAC.x.hdf
    echo ${L2FILE}
    ancfile=${BASE}.L1A_LAC.x.hdf.anc
    echo "Creating GEOFILE $GEOFILE"
    modis_GEO.py -v $FILE -o $GEOFILE
    echo "Creating L1B file"
    modis_L1B.py -v $FILE $GEOFILE
    echo "Creating anc file"
    getanc.py -v $FILE
    echo "Create L2 file"
    l2gen ifile=$L1BFILE geofile=$GEOFILE ofile=$L2FILE par=$ancfile l2prod="Kd_490 Rrs_nnn angstrom aot_869 chlor_a ipar nLw_nnn nflh par pic poc rhos_nnn" aer_opt=-2
    echo "Cleaning up"
    rm -v ${BASE}.L1B* $GEOFILE $ancfile
    #mv -v $FILE done/
    mv -v $L2FILE l2_lac/
done
```

### Things Went Wrong

After running a test data, I found it couldn't run.

Had to reinstall.

And this network was unstable.

![image-20211209155834683](C:/Users/zhenjia/AppData/Roaming/Typora/typora-user-images/image-20211209155834683.png)

Classic 404

After checking, I found their official website was under maintenance.

## Things Went Right Again

```bash
$ bzip2 -d *.bz2
$ ls -1 *L1A_LAC* > input.txt
$ bash l1_to_l2.sh inupt.txt
```
