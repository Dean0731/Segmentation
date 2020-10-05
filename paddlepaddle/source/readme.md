# 安装
  - python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  - python -m pip install paddlepaddle==2.0.0b0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  - pip install paddlex 全流程开发工具
    - 文档 https://paddlex.readthedocs.io/zh_CN/develop/data/format/segmentation.html 
    - cpython
    - pycocotools
        - windows
          - pip install git+https://gitee.com/jiangjiajun/philferriere-cocoapi.git#subdirectory=PythonAPI
          - https://go.microsoft.com/fwlink/?LinkId=691126
        - linux
          - pip install pycocotools
# 模型导出
  - https://paddlex.readthedocs.io/zh_CN/develop/deploy/export_model.html          