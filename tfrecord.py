import tensorflow as tf
import numpy as np
import cv2

# 二進位資料
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 整數資料
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 浮點數資料
def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# 圖片檔案名稱
image_filename = []
# 標示資料
label = []
for i in range(1,11):
    if i < 10:
        for j in range(51,116):
            if j < 100:
                image_filename.append('./training/sample00{}/img00{}-000{}.png'.format(i,i,j))
                t = np.zeros(10)
                t[i-1] = 1.0
                label.append(t)
            else:
                image_filename.append('./training/sample00{}/img00{}-00{}.png'.format(i,i,j))
                t = np.zeros(10)
                t[i-1] = 1.0
                label.append(t)
    else:
        for j in range(51,116):
            if j < 100:
                image_filename.append('./training/sample0{}/img0{}-000{}.png'.format(i,i,j))
                t = np.zeros(10)
                t[i-1] = 1.0
                label.append(t)
            else:
                image_filename.append('./training/sample0{}/img0{}-00{}.png'.format(i,i,j))
                t = np.zeros(10)
                t[i-1] = 1.0
                label.append(t)

print('num of train: ', len(image_filename), ' ', len(label))
# TFRecords 檔案名稱
tfrecords_filename = './training/training.tfrecords'

# 建立 TFRecordWriter
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
print('start to save training')
for image_name, l in zip(image_filename, label):
    # 圖取圖檔
    image = cv2.imread(image_name, 0)

    # 取得圖檔尺寸資訊
    try:
        height, width = image.shape
    except:
        print('no ', image_name)
        continue
    # 序列化資料
    image_string = image.tostring()
    l_s = l.tostring()

    # 建立包含多個 Features 的 Example
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_string': _bytes_feature(image_string),
        'label': _bytes_feature(l_s)}))

    writer.write(example.SerializeToString())

# 關閉 TFRecordWriter
writer.close()
print('training saved')
# 圖片檔案名稱
image_filename = []
# 標示資料
label = []
for i in range(1,11):
    if i < 10:
        for j in range(1,51):
            if j < 10:
                image_filename.append('./validation/sample00{}/img00{}-0000{}.png'.format(i,i,j))
                t = np.zeros(10)
                t[i-1] = 1.0
                label.append(t)
            else:
                image_filename.append('./validation/sample00{}/img00{}-000{}.png'.format(i,i,j))
                t = np.zeros(10)
                t[i-1] = 1.0
                label.append(t)
    else:
        for j in range(1,51):
            if j < 10:
                image_filename.append('./validation/sample0{}/img0{}-0000{}.png'.format(i,i,j))
                t = np.zeros(10)
                t[i-1] = 1.0
                label.append(t)
            else:
                image_filename.append('./validation/sample0{}/img0{}-000{}.png'.format(i,i,j))
                t = np.zeros(10)
                t[i-1] = 1.0
                label.append(t)

print('num of valid: ', len(image_filename))
# TFRecords 檔案名稱
tfrecords_filename = './validation/validation.tfrecords'

# 建立 TFRecordWriter
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
print('start to save training')
for image_name, l in zip(image_filename, label):
  # 圖取圖檔
  image = cv2.imread(image_name, 0)

  # 取得圖檔尺寸資訊
  try:
    height, width = image.shape
  except:
    print('no ', image_name)
    continue

  # 序列化資料
  image_string = image.tostring()
  l_s = l.tostring()

  # 建立包含多個 Features 的 Example
  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'image_string': _bytes_feature(image_string),
      'label': _bytes_feature(l_s)}))

  writer.write(example.SerializeToString())

# 關閉 TFRecordWriter
writer.close()
print('validation saved')