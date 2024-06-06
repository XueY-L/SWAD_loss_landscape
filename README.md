# 仓库介绍
代码是从[SWAD仓库](https://github.com/khanrc/swad/blob/main/visualization/README.md)修改而来，他抄的是[这篇](https://github.com/timgaripov/dnn-mode-connectivity?tab=readme-ov-file)。我们针对DomainNet进行了完善：指定**三个**模型和test_loader，生成loss_landscape等高线图。

# 代码详解
运行losssurface_infer_domainnet.py（命令在文件最上方），生成grid对应的.pth文件。再运行losssurface_plot.py出图