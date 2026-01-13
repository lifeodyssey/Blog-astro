---
title: MODIS水色数据处理教程
copyright: true
tags:
  - Research
  - Ocean Color
  - Oceanography
  - MODIS
  - 大气校正
categories: 学习笔记
mathjax: true
abbrlink: ce08f3a2
date: 2020-04-12 20:09:56
---

这个教程主要分为两部分：

1. 应用python分析MODIS水色遥感数据
2. SeaDAS OCSSW的安装与应用

2021.8.12更新SNAP相关

（最近来看这篇文章的小伙伴好多，可以跟我打个招呼让我认识一下同行么哈哈。

邮箱zhenjiazhou0127@outlook.com，任何问题都可以，欢迎交流）

2022.2.11更新[后续](https://lifeodyssey.github.io/posts/182a5f48.html)

<!-- more -->

# 应用python分析MODIS水色遥感数据

## 程序包及工作环境

如果这个步骤你花了很久才搞定的话，去用R吧，我朋友想跟我学python的都让我劝退学R了。

### Anaconda和 pycharm

Anaconda里涵盖了很多科学计算要用到的包，同时也给了一个很好用的包管理工具。下载地址https://www.anaconda.com/；如果在国内的话，推荐使用清华镜像进行下载https://mirror.tuna.tsinghua.edu.cn/help/anaconda/，然后再更换为国内源https://www.cnblogs.com/yikemogutou/p/11396045.html。

下载完anaconda之后，打开anaconda navigator把Jupyter notebook(或者Jupyter lab)安装好，我个人通常会用这个用一张遥感数据进行程序的调试，然后在pycharm里改成批处理。

pycharm是一个编辑器，同时支持python,R和markdown。对于本菜鸡来说，这个编辑器可以很方便的打开Terminal和python console，添加TODO和利用git备份，最重要的是管理环境方便，不用每次都active啥啥啥。下载地址https://www.jetbrains.com/pycharm/。

### 环境创建

#### 地理信息数据分析环境

打开pycharm，新建一个项目或者打开你之前的项目，然后preference-project interpreter-show all，右下角小加号添加一个python interpreter，选择Conda Environment，用你刚刚安装好的anaconda来创建一个新的环境。

结束之后回到编辑器界面，最下面有四个选项 TODO,Version Control,Terminal,Python Console。这四个东西和刚才那个project interpreter基本是我平时比较喜欢用pycharm的最主要原因了。在写程序时#TODO然后写上要做的事情，就可以在下面TODO页面里看到要做的东西;Version Control 可以提供本地或者github的版本控制，具体设置方法可以参考https://www.jetbrains.com/help/pycharm/manage-projects-hosted-on-github.html；Terminal其实就是一个bash，并且是在你刚才创建的环境里面，不需要去确认环境啥的，也不用添加全局变量，就相当于一个anaconda prompt; python console就相当于一个命令行的python，在右上角绿色小三角附近点下拉菜单Edit configuration，然后选择run in python console，就可以在右边框里看到变量，比较适合我这种从matlab迁移过来的人。

先打开Terminal，依次运行

`conda install -c conda-forge numpy h5py netcd4 opencv-python requests matplotlib pandas scipy scikit-learn`

`conda update -all`

在这里一般不会出问题，-c conda-forge可以安装由大佬在开源社区里上传的程序包

`conda install -c conda-forge pyresample`

这里可能会遇到问题，这里有可能有两种问题，一个是C语言编译器的问题，下载安装Microsoft Visual Studio Community 2017然后再运行一下这句话就好了；另一种就是GDAL的问题。

GDAL(Geospatial Data Abstraction Library)是一个在X/MIT许可协议下的开源栅格空间数据转换库，可以参考大佬翻译的这份中文教程https://www.osgeo.cn/python_gdal_utah_tutorial/index.html来学习一下。之前电脑上没有安装过python相关的东西的话一般不会遇到这个问题。如果你是在windows平台，那么我推荐你来安装OSGeo4W来搞定这个问题https://trac.osgeo.org/osgeo4w/；除此之外，还有两种方法，一个是你重新搞个环境，不要加入之前的包再来安装一次，另一个是下载whl文件来安装。

记得每次装完新的都来个conda update -all。

最后

`conda install -c conda-forge basemap basemape-data-hires`

 `conda install -c conda-forge gdal`

`conda update -all`

如果一切顺利，恭喜你完成了最麻烦的一步。

之后可以参照我之前写过的这篇文章里面的python包，https://lifeodyssey.github.io/post/3aa0ed1a.html ,这些包都安装之后基本够用了。

其实环境这个问题可以用Docker来解决，之前有看到过这个大佬https://zhuanlan.zhihu.com/p/108012664的文章，我也还在研究，争取尽快搞出来一个水色人的Docker。

#### pycharm及Jupyter插件

这个是属于个人喜好。

在pycharm-preference-plugins里可以安装插件，我安装了.*ignore,Dart,Kite和Material Theme UI，因为我有的时候还会用R，虽然R studio更好用，但是我也装了R Language for IntelliJ这个，来进行一些小的调试。

Jupyter个人一般只用来学习新的知识、调试和改bug，或者用来教别人的时候用，所以我这里用的还是Jupyter notebook,没有升级到Jupyterlab，安装教程可见https://github.com/ipython-contrib/jupyter_contrib_nbextensions。

在Terminal里依次输入

`conda install -c conda-forge jupyter_contrib_nbextensions`

`jupyter contrib nbextension install --user`

`conda update -all`

然后在Terminal里输入

`Jupyter notebook`

就可以看到浏览器里蹦出来一个Jupyter notebook

点开Configurable nbextensions中勾选variable inspector，然后随便新建一个notebook，点开那个小瞄准镜就可以看到变量了。我平时没事就只用这个了。

## MODIS 数据读取、重投影及绘图

数据的下载可以参考https://lifeodyssey.github.io/post/8636bca2.html

直接上我写的代码，造福千万家，但是希望你直接用的时候能够知其然也知其所以然。感谢实验室之前毕业的前辈给我提供的代码样本

```python
import netCDF4 as nc4
import numpy as np
from collections import OrderedDict
from geo_Collection import geo_web as gs
from QAAV6 import QAAv6
minlat = 32.5
minlon = 130.5
maxlat = 35
maxlon = 136
# area of full seto-inland sea
x = np.arange(minlon, maxlon, 0.01)  # 1 km grid,
y = np.arange(maxlat, minlat, -0.01)
nc_file = nc4.Dataset(filename, 'r')
lon = nc_file.groups['navigation_data'].variables['longitude'][:]
lat = nc_file.groups['navigation_data'].variables['latitude'][:]
variables = nc_file.groups['geophysical_data'].variables
for i in variables:
    var = variables[i][:]
    np.where(var <= 0, var, np.nan)
    if i != 'l2_flags':
        var_re, grid = gs.swath_resampling(var, lon, lat, x, y, 5000)  # 1 km grid
        # var_re=var_re.filled()
        if np.ma.is_mask(var_re):
            var_re.mask=np.ma.nomask
            var_re[var_re==-32767.0]=np.nan
         variables[i] = var_re
     else:
            # var_re = var_re.filled()
          variables[i] = var
lons = grid.lons
lats = grid.lats
#以叶绿素为例
chl = variables['chlor_a']
plot_geo_image(chl, lon, lat, label='CHL [mg/m$^3$]',
               title=os.path.basename(file))
ncfile.close()#一定要记得这句



```

同时给出这里面用到的两个子函数和QAA的代码。

（QAA的代码还需要一些改进）

```python
import pyresample
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import colors
import math
from scipy.interpolate import RectSphereBivariateSpline
from matplotlib.colors import LogNorm
#from  deco import *
def swath_resampling(src_data: np.ma.array, src_lon: np.array, src_lat: np.array,
                     trg_lon: np.array, trg_lat: np.array, search_radius: float):
    if len(trg_lon.shape) == 1:
        grid_def = pyresample.geometry.SwathDefinition(*np.meshgrid(trg_lon, trg_lat))
    else:
        grid_def = pyresample.geometry.SwathDefinition(lons=trg_lon, lats=trg_lat)

    #source grid with original swath data
    # if len(src_lon.shape) == 1:
    #     swath_def = pyresample.geometry.SwathDefinition(*np.meshgrid(src_lon, src_lat,sparse=True))
    # else:
    #     swath_def = pyresample.geometry.SwathDefinition(lons=src_lon, lats=src_lat)

    swath_def = pyresample.geometry.SwathDefinition(lons=src_lon, lats=src_lat)
    # resample (here we use nearest. Bilinear, gaussian and custom defined methods are available)
    # for more, visit https://pyresample.readthedocs.io/en/latest/
    result = pyresample.kd_tree.resample_nearest(swath_def, src_data, grid_def, epsilon=0.5,
                                                 fill_value=np.nan, radius_of_influence=search_radius)
    return result, grid_def

#@concurrent
def plot_geo_image(sds: np.ma.array, lon: np.ndarray, lat: np.ndarray, log10: bool = True, title: str = None,
                   label: str = None,
                   caxis: list = None, lon_range: list = None, lat_range: list = None, save_image: str = None,
                   dpi: int = 400):
    if len(lon.shape) == 1:
        print('MeshGridding...')
        lon, lat = np.meshgrid(lon, lat)

    lon_0 = (lon.min() + lon.max()) / 2
    lat_0 = (lat.min() + lat.max()) / 2

    print(f'Lat: [{lat.min():.3f}, {lat.max():.3f}] | '
          f'Lon: [{lon.min():.3f}, {lon.max():.3f}] | '
          f'SDS: [{sds.min():.3f}, {sds.max():.3f}]')

    if (lon_range is not None) and (lat_range is not None):
        m = Basemap(llcrnrlon=min(lon_range), llcrnrlat=min(lat_range),
                    urcrnrlon=max(lon_range), urcrnrlat=max(lat_range),
                    resolution='f', lon_0=lon_0, lat_0=lat_0, projection='tmerc')
    else:
        m = Basemap(llcrnrlon=lon.min(), llcrnrlat=lat.min(),
                    urcrnrlon=lon.max(), urcrnrlat=lat.max(),
                    resolution='f', lon_0=lon_0, lat_0=lat_0, projection='tmerc')
    x2d, y2d = m(lon, lat)

    fig = plt.figure(figsize=(8, 8 * m.aspect))
    ax = fig.add_axes([0.08, 0.1, 0.7, 0.7], facecolor='white')
    # changed to facecolor 8 October 2019

    if (lon_range is not None) and (lat_range is not None):
        parallels = np.arange(min(lat_range), max(lat_range), 3)
        meridians = np.arange(min(lon_range), max(lon_range), 4)
    else:
        parallels = meridians = None

    if caxis is not None:
        cmn, cmx = min(caxis), max(caxis)
    else:
        cmn, cmx = sds.min(), sds.max()
    # m.drawparallels(parallels, fontsize=10, linewidth=0.25, dashes=[7, 15],
    #                  color='k', labels=[1, 0, 1, 1])
    # m.drawmeridians(meridians, fontsize=10, dashes=[7, 15],
    #                  linewidth=0.3, color='k', labels=[1, 1, 0, 1])
    #ncl = 150
    #if log10 is True:
    #    norm = colors.LogNorm(vmin=cmn, vmax=cmx)
    #else:
     #   bounds = np.linspace(cmn, cmx, ncl)
     #   norm = colors.BoundaryNorm(boundaries=bounds, ncolors=ncl)

    p = m.pcolor(x2d, y2d, sds, vmin=cmn,vmax=cmx, cmap='jet')

    if title is not None:
        plt.title(title)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('vertical', size="3%", pad=0.05)
    #cax = plt.axes([cmn, 0, cmx])  # setup colorbar axes

    cb = m.colorbar(p,location="right",size="5%",pad=0.1)  # draw colorbar
    #if label is not None:
    # cb.set_label("%s" % label)
    # plt.sca(ax)  # make the original axes current again
    # plt.clim(cmn, cmx)
    #unit='Elevation to the sea level'
    #cb.set_label(unit, rotation=270, labelpad=10.0, fontsize=10)
    cb.ax.tick_params(labelsize=10)

    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents()
    #plt.show()

    if save_image is not None:
        plt.savefig(save_image, dpi=dpi, facecolor='w', edgecolor='w', orientation='portrait')
        plt.show()
        plt.close()

def creategrid(min_lon, max_lon, min_lat, max_lat, cell_size_deg, mesh=False):
#Output grid within geobounds and specifice cell size
#cell_size_deg should be in decimal degrees’’’

    min_lon = math.floor(min_lon)
    max_lon = math.ceil(max_lon)
    min_lat = math.floor(min_lat)
    max_lat = math.ceil(max_lat)
    lon_num = (max_lon - min_lon)/cell_size_deg
    lat_num = (max_lat - min_lat)/cell_size_deg
    grid_lons = np.zeros(lon_num) # fill with lon_min
    grid_lats = np.zeros(lat_num) # fill with lon_max
    grid_lons = grid_lons + (np.assary(range(lon_num))*cell_size_deg)
    grid_lats = grid_lats + (np.assary(range(lat_num))*cell_size_deg)
    grid_lons, grid_lats = np.meshgrid(grid_lons, grid_lats)
    grid_lons = np.ravel(grid_lons)
    grid_lats = np.ravel(grid_lats)
    #if mesh = True:
    # grid_lons = grid_lons
    # grid_lats = grid_lats
    return grid_lons, grid_lats
```

QAAv6

```python
from scipy.interpolate import Akima1DInterpolator as Akima
import numpy as np
from deco import *
import math as m


# wavelengths = {
#     'OLI'   : [442.98, 482.49, 561.33, 654.61],
#     'MSI'   : [443.93, 496.54, 560.01, 664.45],
#     'OLCI'  : [411.3999939, 442.63000488, 490.07998657, 510.07000732, 560.05999756, 619.97998047, 664.85998535, 673.61999512, 681.15002441], # 9 band insitu
#     'OLCI2' : [400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25, 708.75, 753.75, 761.25, 764.375, 767.5, 778.75], # 16 band LUT
#     'VI'    : [412.49, 444.17, 486.81, 549.99, 670.01],
#     'AER'   : [412, 442, 490, 530, 551, 668],
#     'MOSIA' : [412, 443, 469,488,531,547,555,645,667,678],
#     'GOCI'  : [412, 443, 490, 555, 660, 680],
#     'SGLI'  : [380,412,443,490,530,565,673.5]
#}
#@concurrent
def QAAv6(Rrs):
    #acording to Lee,QAAv6
    #write by Zhou,20191130
    #Input data need to be an arrary that contain 10 bands Rrs of MODIS,from short wavelength to long wavelength in a certain station
    #Output is a tuple, first array is aph,second array is bbp
    #use as import QAAV6
# B1: 412 nm 0
# B2: 443 nm 1
# B3: 469 nm 2
# B4: 488 nm 3
# B5: 531 nm 4
# B6: 547 nm 5
# B7: 555 nm 6
# B8: 645 nm 7
# B9: 667 nm 8
# B10: 678 nm 9

    Lambda=np.array([412, 443, 469,488,531,547,555,645,667,678])
    nbands=np.shape(Lambda)[0]

    IOPw=np.array([[0.003344468,0.004572564],
    [0.00244466,0.00635],
    [0.001910803,0.010483637],
    [0.001609567,0.014361745],
    [0.00111757,0.043747657],
    [0.000983055,0.053262848],
    [0.000923288,0.0595],
    [0.000482375,0.325],
    [0.00041731,0.433497423],
    [0.00038884,0.457440162]])

    #     if(Rrs=np.nan):
    #     return np.nan
    # else:
    # bbw from Morel (1974).aw  from Pope and Fry (1997)
    bbp = np.ones(10)
    adg = np.ones(10)
    if(np.nan in Rrs):
        bbp[:]=np.nan
    else:
        bw=IOPw[:,0]#backscaterring of pure water
        aw=IOPw[:,1]#absorption of pure water
        rrs = Rrs / (0.52 + 1.7 * Rrs)
        g0 = 0.089
        g1 = 0.1245
        u = (-g0 + ((g0 ** 2) + 4 * g1 * rrs) ** 0.5) / (2 * g1)

        aph = np.ones(10)#adg is the absorption of CDOM and NAP
        if Rrs[6]<0.0015:#select 555 as reference
            r=550
            p1=(rrs[1] + rrs[3])
            p2 = rrs[6] + 5 * (((rrs[8]) ** 2)) / (rrs[3])
            x = np.log10(p1 / p2)
            ar = aw[6] + np.power(10, (-1.146 - 1.366 * x - 0.469 * (x ** 2)))# step 2
            bbpr=((u[6]*ar)/(1-u[6]))-bw[6]#step3
        else:
            r=670
            p1 = Rrs[8] / (Rrs[1] + Rrs[3])
            p2 = 0.39 * (p1 ** 1.14)
            ar = (aw[8]) + p2  # step2
            bbpr = (u[8] * ar / (1 - (u[8])) - (bw[8]))  # step3
        eta=2*(1-1.2*np.exp(-0.9*(rrs[1]/rrs[6]))) #step4

        zeta = 0.74 + 0.2 / (0.8 + rrs[1] / rrs[6])#step 7&8
        S = 0.015 + 0.002 / (0.6 + rrs[1] / rrs[6])
        xi = np.exp(S * (442.5 - 415.5))
        for i in range(nbands):
            bbp[i]= bbpr * np.power(r/Lambda[i], eta)#step5
        a = ((1 - u) * (bw + bbp)) / u#step6
        for i in range(nbands):

            ag443=((a[0]-zeta*a[1])/(xi-zeta))-((aw[0]-zeta*aw[1])/(xi-zeta))
            adg[i]=ag443*np.exp(-S*(Lambda[i]-443))
            aph[i]=a[i]-adg[i]-aw[i]
        return bbp,a

```

QAA的代码还需要很多的改进，目前正在参考https://github.com/BrandonSmithJ/band-adjustment进行改进，欢迎同行交流。

## 程序加速

因为python自己的问题，如果你要处理大量遥感图像的话会很慢，这个一方面是matplotlib自己出图慢，另一方面是因为没有充分利用多核cpu。

批量出图可以参考https://cloud.tencent.com/developer/article/1584962 这篇文章。

解决多核cpu最好用的方法是并行。

像上面那个程序，用并行来加速的话可以这么写：

1. 把它改写成一个函数，一般是把文件名作为输入，或者是文件在datalist之中的顺序，我用的是后面这种。
2. a1=np.arange(len(datalist))，创建一个list，并行的时候传入的变量只能是list
3. 将代码改写成如下样子

```python
import multiprocessing as mp
def main()
#这里是你刚才改写的程序
pool=mp.Pool(processes=7)#电脑8核，给自己空余了一个核用来处理其他任务
pool.map(main,a1)

```

可以先不做并行查看内存占用情况，如果有很多内存没用的话，开了并行一般都会提速，我的程序需要处理140张卫星图像，缩短到原来的1/3左右。

# SeaDAS OCSSW的安装与应用

SeaDAS是NASA出品的一个水色遥感处理软件，其中的OCSSW更是内置了很多美国发射的卫星的标准处理流程算法（欧洲的那几个用的SNAP，韩国的GOCI用的GDPS，日本的SGLI，也许哪天我用OLCI数据的时候会把SNAP的教程搞一下），我主要是用他的l2gen这个功能。OCSSW只能在Linux或者Mac OS系统下安装。下载地址https://seadas.gsfc.nasa.gov/，这里讲Linux系统下的安装方法，我用的是Ubuntu 18.04LTS.

## 安装

如果你像我一样肉身翻墙的话，那恭喜你真的是非常方便了。

### SeaDAS 安装

NASA官方的tutorial在https://seadas.gsfc.nasa.gov/tutorials/installation_tutorial/

基本按照这个来就可以了，除了Java版本，我第一次装的时候用的是最新版Java，结果SeaDAS无法作为一个独立程序运行（就是在程序页面里找不到这个程序的图标？启动器？本Ubuntu菜鸡虚心求教），最后按照如下方法安装了Java8:

```bash
sudo add-apt-repository ppa:ts.sch.gr/ppa 
sudo apt-get update 
sudo apt-get install oracle-java8-installer
```

### OCSSW安装

#### Python环境设置

我们一般现在用来干活的python一般都是3版本的，但是Ubuntu系统内置的python是2版本，SeaDAS OCSSW会使用默认的内置版本，所以在这里需要搞定一些程序库的问题（如果没搞好的话后面一般会显示你没有requests这个包）。

这时候就用到我刚才安装的Pycharm来手动管理了，打开pycharm-preference-Project Interpreter,如果你没有装别的东西的话，你的电脑里应该就只有几个python的解释器，你刚才通过anacoonda安装的是3版本，在Add Python Interpreter里面，系统自带的解释器一般会在Pipenv、System或者Virtualenv这里找到，通过路径判断出来哪个是系统自带的，创建一个新环境，然后在Terminal里安装requests这个包就可以了

#### 安装

完成这一步之后，如果你肉身翻墙或者依靠金钱的力量翻墙的话，就可以在GUI里面手动点选安装了；如果在国内，可以按照刚才放的官方的页面，把每个Bundle都下载下来离线安装。

## 应用

我这里主要用的是l2gen这个函数，这里放一个在seadas forum找到然后自己修改的脚本

```bash
# source /Users/zhenjia/.bash_profile 
#先运行上面这个文件来搞定全局变量
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
    BASE=`basename $FILE .L1A_LAC.x.hdf`
    echo ${BASE}
    GEOFILE=${BASE}.GEO
    echo ${GEOFILE}
    L1BFILE=${BASE}.L1B_LAC.x.hdf
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

# SNAP

最近ESA的SNAP宣布了对SeaDAS OCSSW的支持，这个不需要搞Linux系统啥的了，用起来方便很多。

下载地址：https://step.esa.int/main/download/snap-download/

下载安装之后在[这里](https://seadas.gsfc.nasa.gov/installers/snap-seadas-toolbox/)下载SeaDAS的插件。然后打开SNAP，打开Plugin Manager，在菜单中依次选择Tools --> Plugins-->Downloaded--> Add Plugins，定位到你解压好的插件的位置，选择好之后点Open，然后Install->finish即可。

顺带一提，SNAP还支持最近几个比较新的大气校正和IOP反演的算法插件，比如[OC-SMART](http://www.rtatmocn.com/oc-smart/), [3SAA](https://gitlab.eumetsat.int/eumetlab/oceans/ocean-science-studies/olci-iop-processor/-/tree/master)。

