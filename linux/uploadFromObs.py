from modelarts.session import Session
session = Session()
session.download_data(bucket_path="voc2012/dataset/dom/segmentation4/", path="/home/ma-user/work/segmentation4/")

