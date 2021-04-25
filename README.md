# SISR模型框架

## 文件结构

该框架的文件夹结构如下：

```
|-- SR_framework
	|-- config
	|-- data
		|-- train
		|-- test
	|-- demo_input
	|-- demo_output
	|-- log
	|-- models
	|-- test_result
	|-- trained_model
	|-- utils
```

其中，每个文件夹分别实现以下功能：

`config`：存放模型的超参数文件

`data`：存放数据集，其中`train`为训练集存放位置，`test`为测试集存放位置

`demo_input`：用于存放自定义演示图像的文件夹

`demo_output`：演示输出图像文件夹

`log`：存放日志文件

`models`：存放模型定义文件的位置，其模型的调用通过`__init__.py`实现

`test_result`：存放测试结果的位置，其中每个模型的测试结果会放在其相应的文件夹下

`trained_model`：存放训练后的参数文件的位置，其中每个模型的参数文件会放在其相应的文件夹下

`utils`：存放实现框架所需的函数、类和方法等

## 准备工作

### 环境配置

本框架使用`Matlab2019`和`python 3.6`以上环境。

本框架需要以下python包：

```
random
torch
math
json
numpy
torchvision
datetime
tqdm
opencv-python
pandas
matplotlib
copy
```

### 首次运行

如果你是第一次使用这个框架，你可能会面临文件夹不存在的情况，因此需要运行`check_folder.py`文件，它会自动扫描当前目录下的文件夹，并创建缺失的文件夹。

### 数据集

将训练集文件夹拷贝至`./data/train`文件夹内，然后将测试集的一系列文件夹拷贝至`./data/test`文件夹内。若采取多个训练集同时训练，需要将多个训练集的图片放在一个文件夹中。

然后在Matlab中打开`prepare_LR.m`文件，在第四行`model_mode`更改选择模型框架，`'pre'`为预上采样SR框架，`'post'`为后上采样SR框架，需要注意的是两种框架对应的LR尺寸不同；在第五行的下采样模式`downsample_mode`中更改下采样方法，`'BI'`为仅采用双三次插值法，`'BD'`为高斯模糊下采样；在第六行选择模型训练所使用的色彩通道，`RGB`为采用彩色通道，`Y`为采用亮度通道。运行文件生成对应的LR图像。

注意：Matlab运行时需要将工作文件夹设置为本框架的根目录。

### 模型定义

将模型的定义文件拷贝至`model`文件夹内。然后通过`from .a import b`方法导入相关的模型，`a`处填写模型的文件名，`b`处填写模型文件中定义的模型类名。之后在`model_list`字典中填写你所命名的模型名`'c'`，写上冒号，然后写上`b`处的模型类名，即：`'c': b`，这样就可以通过传递`'c'`来调用模型。

如果我们的模型叫`MSRN`，模型文件定义中的模型类也叫`MSRN`，就可以写成`'MSRN': MSRN`。

在这个框架中，模型以上采样位置被分为预上采样模型和后上采样模型，在后上采样模型中按照放大系数的定义方法可以分为在模型初始化时定义和在前向传播时定义。预上采样模型和在前向传播时定义放大系数的后上采样模型默认不需要在初始化时传入任何参数；但是，模型初始化时定义放大系数的后上采样模型必须要传入放大系数参数，而且这个参数的名称必须是`scale`，否则就会报错。假如我们的模型叫做MSRN，这是一个需要在模型初始化时传入放大系数的后上采样模型，在写这个模型类时候就需要在初始化时传入其放大系数参数，且这个参数的名称必须为`scale`，如下所示：

```
class MSRN(nn.Module):
    def __init__(self, scale):
    	······
```

### 模型超参数json文件设置

模型的超参数文件保存在config文件夹下，其格式为json文件，文件名必须要和之前的模型的命名一致，即文件名要和上面的`c`一样。该文件下需要要有以下几个大项目：

`system`：主要的一些超参数

`learning_rate`：学习率的相关设置

`dataloader`：数据加载器(Dataloader)的相关设置

`test`：对模型的测试的相关设置

以下是每个参数的详细解释：

#### system

`model_name`：模型名称，可以和上面的命名不一样，这个将作为训练和验证的模型名，且训练后的模型参数文件和测试的结果将会保存在以这个为名的文件夹下。

`dataset`：训练集，名称需要与`./data/train/`文件夹下的训练集名称一致

`model_mode`：模型所采用的SR框架，即该模型是预上采样模型还是后上采样模型，`'pre'`为预上采样，`'post'`为后上采样。在这边，只要模型输入经过双三次插值处理后的LR图像统一设置为`'pre'`；只要模型输出图像尺寸比输入大的统一设置为`'post'`

`color_channel`：模型的色彩通道，若模型输入为3通道的`RGB`图像，则设置为`RGB`；若模型输入为1通道的黑白图像，则设置为`Y`

`batch_size`：批大小

`patch_size`：LR的切片大小，将LR和HR随机切片为一定大小的小块。在预上采样模型中，LR和HR切片大小一致；在后上采样模型中，HR的切片大小为`scale * patch_size`

`Epoch`：遍历数据集的轮数

`device`：训练和验证所使用的设备，默认为`cuda:0`

`scale_factor`：放大系数，可以设置为一个单一值，也可以设置成一个数组。设置为数组时模型将按照数组中的放大系数依次训练

`save_step`：每遍历多少次数据集保存一次模型

`weight_init`：参数初始化方法，其中可选无、Xavier方法和标准化方法，即`None`、`Xavier`和`Normal`

`loss_function`：损失函数，目前可选`L1`、`L2`和`L1_Charbonnier`三种

`optimizer`：优化器，目前可选`SGD`和`Adam`两种

`scale_position`：模型中对放大系数定义的位置，这个在后上采样模型中可用，`init`表示在模型定义时就需要固定放大系数，放大系数信息会传入到模型中并固定下来；`forward`表示在前向传播时输入放大系数，模型定义时并不需要固定放大系数，只有训练和验证时前向传播阶段才传入放大系数信息

`model_args`：用于调整模型结构的超参数，默认为无，若有必要的话可以对其设置，并需要与模型类对得上

`loss_args`：用于设置损失函数的超参数，默认为无，若有必要的话可以对其设置，并需要与损失函数传参对得上

`optim_args`：用于设置优化器的参数，默认为无，但是需要于优化器传参对得上

#### learning_rate

`init_learning_rate`：初始学习率

`decay_mode`：学习率衰减方法，可以选择无、固定步长衰减、指数衰减、多步长衰减和余弦退火衰减，即`None`、`Step`、`Exp`、`MultiStep`和`CosineAnnealing`。

​	当选择`None`时不需要定义之后的任何参数；

​	当选择`Exp`时需要定义`decay_rate`参数；

​	当选择`Step`和`MultiStep`时需要定义以下的`per_epoch`和`decay_rate`两个参数；

​	当选择`CosineAnnealing`时需要定义`per_epoch`和`eta_min`两个参数

`per_epoch`：每遍历数据集多少轮后执行学习率衰减。在使用`Step`、和`CosineAnnealing`时需要填写一个整数；在使用`MultiStep`时需要填写一个Epoch的列表

`decay_rate`：衰减率，需要设置为一个小于1的数，在指数衰减时表示`gamma`值

`eta_min`：学习率的最小值，默认为0

#### dataloader

`num_workers`：加载器所使用的线程数

`shuffle`：是否打乱数据集，设为`True`表示打乱，设为`False`表示不打乱

`drop_last`：是否丢弃最后一批数量不足batch_size的数据，默认为`False`

`pin_memory`：是否使用锁页内存，默认为`True`，若你的内存较小则可以选择为`False`

`normalize`：是否使用归一化，若为`True`则将所有图像像素归一化至`[0, 1]`之间；若为`False`则不执行归一化，图像像素会在`[0, 255]`之间

#### test

`color_channel`：测试的色彩通道，可以使用RGB通道或者YCbCr中的亮度通道进行测试，即`RGB`和`Y`。若模型只针对图像的Y通道进行训练的话则只能选择`Y`

`drew_pic`：是否绘制模型测试曲线图，仅在对所有模型测试时使用，调用后会绘制每一个保存的模型文件的指标图

`test_dataset`：测试集，用一个`list`表示，其中每一个测试集的名称都要与`./data/test/`中的测试集名称匹配，测试时会按照这个依次进行测试

`shave`：测试时每张图像从边缘裁剪的像素大小，可以直接设置数字，或者设置为`"scale"`，表示和当前放大系数相同，默认为0

## 训练

训练模型只需要调用`train.py`即可，然后在后面写上模型的名字，即：

```
python train.py <model>
```

其中，`model`的名字必须和config中的json文件以及models文件夹中的模型类名一致，否则会报错

例：

```
python train.py SRCNN
```



若你的模型跑到一半崩了，那么可以用断点模式对上一次保存的模型文件继续训练，如下：

```
python train.py <model> --breakpoint 'para file'
```

其中，`'para_file'`格式如下：`'net_x3_50.pth'`，这个是保存在`train_model/(model_name)/`下的模型参数文件，调用的时候得确保这个文件是存在的，否则会报错

例：

```
python train.py SRCNN --breakpoint 'net_x2_100.pth'
```

## 测试

测试模型只需要调用`test.py`即可，然后在后面写上模型的名字、测试模式和模型文件，如果你只想测试某一个数据集，那么需要写上数据集文件名，即：

```
python test.py <model> --<mode> 'para_file' --dataset 'dataset name'
```

其中，`mode`参数可以选择`all`和`once`。选择`all`会对当前模型的所有参数文件进行测试，可用于模型的收敛性测试或者找出效果最好的模型文件；选择`once`则对`'para_file'`的模型文件进行测试，`'para_file'`的格式和训练部分相同，不过可以测试`.pkl`文件，需要确保这个文件是存在的，否则会报错

`dataset`是一个可选参数，如果不选择参数，模型会按照`json`文件的测试部分对`test_dataset`遍历并逐一测试，若选择该参数，则会只对`'dataset name'`执行测试，需要注意的是`'dataset name'`必须存在于`./data/test/`中，否则会报错

例：

```
python test.py SRCNN --all
```

```
python test.py SRCNN --once 'net_x2_150.pth'
```

```
python test.py SRCNN --all --dataset 'Set5'
```

```
python test.py SRCNN --once 'net_x2_150.pth' --dataset 'Set14'
```

测试的所有结果保存在`./test_result/(model_name)/`文件夹下

## 演示模式

想查看模型实际的效果，可以通过运行`demo.py`实现，在该模式下需要写上模型的名字、保存的模型文件名、需要演示的数据集等，如下所示：

```
python demo.py <model> --file 'para_file' --<data> 'dataset name'
```

其中，`file`参数是你需要演示的模型文件或模型参数文件，`para_file`格式和测试部分相同

`data`参数可以选择`dataset`或`input`。`dataset`表示对某一个测试集进行演示，在这个模式下需要添加`'dataset name'`，其格式于测试部分相同；若选择`input`模式，则会对`demo_input`文件夹中的所有图像进行演示

例：

```
python demo.py SRCNN --file 'net_x3_100.pth' --input
```

```
python demo.py SRCNN --file 'net_x3_100.pth' --dataset 'Set5'
```

演示的所有图像保存在`./demo_output/(model_name)/`文件夹下

**注意：在使用自定义图像演示预上采样模型时需要先将图像放大至相应的放大系数**

## 版本迭代

v0.1 内部测试版本

v0.2 修复bug，增加json文件的模型参数和损失函数参数功能

v0.3 修复bug，将测试部分的测试模式从json文件转移至命令行参数

v0.4 修复bug，添加测试指定数据集功能

v0.5 修改训练图像色彩通道逻辑，提升速度

v0.6 修复bug，添加json文件的优化器参数功能

v0.7 添加测试指定模型时将结果绘制成表格的功能

v0.8 添加演示模式和检查文件夹功能

v1.0 发布版本