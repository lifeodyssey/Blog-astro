---
title: FY4A数据处理
tags:
  - 国产卫星数据处理
  - FY4A
categories: 学习笔记
abbrlink: d464ecc8
password: 
date: 2021-12-13 14:17:33
mathjax:
copyright:
---
工作原因接了个FY4A数据处理的活
<!-- more -->

我想这这种东西肯定有现成的轮子，果不其然让我在gayhub上找到了[这个](https://github.com/Mo-Dabao/BiteFengyun)

# 行列转换

源代码这么写的

```bash
from numpy import deg2rad, rad2deg, arctan, arcsin, tan, sqrt, cos, sin


ea = 6378.137  # 地球的半长轴[km]
eb = 6356.7523  # 地球的短半轴[km]
h = 42164  # 地心到卫星质心的距离[km]
λD = deg2rad(104.7)  # 卫星星下点所在经度
# 列偏移
COFF = {"0500M": 10991.5,
        "1000M": 5495.5,
        "2000M": 2747.5,
        "4000M": 1373.5}
# 列比例因子
CFAC = {"0500M": 81865099,
        "1000M": 40932549,
        "2000M": 20466274,
        "4000M": 10233137}
LOFF = COFF  # 行偏移
LFAC = CFAC  # 行比例因子


def latlon2linecolumn(lat, lon, resolution):
    """
    (lat, lon) → (line, column)
    resolution：文件名中的分辨率{'0500M', '1000M', '2000M', '4000M'}
    line, column不是整数
    """
    # Step1.检查地理经纬度
    # Step2.将地理经纬度的角度表示转化为弧度表示
    lat = deg2rad(lat)
    lon = deg2rad(lon)
    # Step3.将地理经纬度转化成地心经纬度
    eb2_ea2 = eb**2 / ea**2
    λe = lon
    φe = arctan(eb2_ea2 * tan(lat))
    # Step4.求Re
    cosφe = cos(φe)
    re = eb / sqrt(1 - (1 - eb2_ea2) * cosφe**2)
    # Step5.求r1,r2,r3
    λe_λD = λe - λD
    r1 = h - re * cosφe * cos(λe_λD)
    r2 = -re * cosφe * sin(λe_λD)
    r3 = re * sin(φe)
    # Step6.求rn,x,y
    rn = sqrt(r1**2 + r2**2 + r3**2)
    x = rad2deg(arctan(-r2 / r1))
    y = rad2deg(arcsin(-r3 / rn))
    # Step7.求c,l
    column = COFF[resolution] + x * 2**-16 * CFAC[resolution]
    line = LOFF[resolution] + y * 2**-16 * LFAC[resolution]
    return line, column


def linecolumn2latlon(line, column, resolution):
    """
    (line, column) → (lat, lon)
    resolution：文件名中的分辨率{'0500M', '1000M', '2000M', '4000M'}
    """
    # Step1.求x,y
    x = deg2rad((column - COFF[resolution]) / (2**-16 * CFAC[resolution]))
    y = deg2rad((line - LOFF[resolution]) / (2**-16 * LFAC[resolution]))
    # Step2.求sd,sn,s1,s2,s3,sxy
    cosx = cos(x)
    cosy = cos(y)
    siny = sin(y)
    cos2y = cosy**2
    hcosxcosy = h * cosx * cosy
    cos2y_ea_eb_siny_2 = cos2y + (ea / eb * siny)**2
    sd = sqrt(hcosxcosy**2 - cos2y_ea_eb_siny_2 * (h**2 - ea**2))
    sn = (hcosxcosy - sd) / cos2y_ea_eb_siny_2
    s1 = h - sn * cosx * cosy
    s2 = sn * sin(x) * cosy
    s3 = -sn * siny
    sxy = sqrt(s1**2 + s2**2)
    # Step3.求lon,lat
    lon = rad2deg(arctan(s2 / s1) + λD)
    lat = rad2deg(arctan(ea**2 / eb**2 * s3 / sxy))
    return lat, lon


# if __name__ == "__main__":
#     """
#     调用示范
#     """
#     from numpy import arange, meshgrid, concatenate, float32, newaxis
#     # 设置插值步长、经纬度范围
#     interp_steps = {"0500M": 0.005,
#                     "1000M": 0.01,
#                     "2000M": 0.02,
#                     "4000M": 0.04}
#     lat_S, lat_N = 0, 50
#     lon_W, lon_E = 80, 130
#     # 先乘1000取整是为了防止浮点数的精度误差累积
#     lat_S = int(1000 * lat_S)
#     lat_N = int(1000 * lat_N)
#     lon_W = int(1000 * lon_W)
#     lon_E = int(1000 * lon_E)
#     interp_steps = {x: int(y * 1000) for x, y in interp_steps.items()}
#     # 开始经纬度转行列号（内存充足情况下）
#     lc = {}  # 保存各分辨率经纬度对应的行列号为字典
#     for resolution, interp_step in interp_steps.items():
#         lat = arange(lat_N, lat_S-1, -interp_step) / 1000
#         lon = arange(lon_W, lon_E+1, interp_step) / 1000
#         lat = lat.astype(float32)
#         lon = lon.astype(float32)
#         lon, lat = meshgrid(lon, lat)  # 构造经纬度网格
#         line, column = latlon2lc(lat, lon, resolution)
#         l = l[:, :, newaxis]
#         c = c[:, :, newaxis]
#         lc[resolution] = concatenate((l, c), axis=2)
```

写的真好.jpg

原来那个库的另一部分是做L1的辐射定标，其实satpy也支持那个，我就不多讲了。

# 重投影画图

这个才是我要搞得。

虽然上边那个很方便，但是对我来讲搞起来就太麻烦了，我每次搞投影都是重新下一次。

就直接在[这里](http://satellite.nsmc.org.cn/PortalSite/StaticContent/DocumentDownload.aspx?TypeID=9)下了查找表

下完解压之后

```python
def fy4disk(rawfile,dim):
    sz = np.fromfile(rawfile, dtype=float, count=dim*dim*2)
    latlon = np.reshape(sz,(dim,dim,2))

    lat = latlon[:,:,0]
    lon = latlon[:,:,1]

    lat[lat > 100] = -9999.
    lon[lon < 0  ] = lon[lon < 0  ] + 360.
    lon[lon > 361] = -9999.

    return lon, lat

def fy4disk(rawfile,dim):
    sz = np.fromfile(rawfile, dtype=float, count=dim*dim*2)
    latlon = np.reshape(sz,(dim,dim,2))

    lat = latlon[:,:,0]
    lon = latlon[:,:,1]

    lat[lat > 100] = np.nan
    lon[lon < 0  ] = lon[lon < 0  ] + 360.
    lon[lon > 361] = np.nan

    return lon, lat

rawfile = r'H:\gOOLE\FullMask_Grid_4000\FullMask_Grid_4000.raw'


dim = 2748 # 4km

    # Two ways of reading an binary file
# lon, lat = fy4raw(rawfile,dim)
lon, lat = fy4disk(rawfile,dim)


SST=np.array(SST)
SST=np.ma.masked_where(SST<=-5,SST)
SST=np.ma.masked_where(SST>=45,SST)
fig = plt.figure(figsize=(16, 16))  # 设置画布
plt.axis('off')
proj = ccrs.PlateCarree()  # 创建一个投影
ax = plt.axes(projection=proj)  # 创建一个画纸， 并指明投影类型
extent = [70, 140, 0, 60]  # 显示范围
ax.set_extent(extent, proj)
cs = ax.contourf(lon, lat, SST, transform=proj, cmap='rainbow')
ax.background_patch.set_visible(False)  # Background
ax.outline_patch.set_visible(False)  # Borders
plt.savefig(os.path.splitext(f1)[0],dpi=300, facecolor='w', edgecolor='w', orientation='portrait',transparent=True)
```

