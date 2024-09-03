# 根据灰度值划分图片图片
import os  
from PIL import Image  
import numpy as np
  
def is_image_file(filename):  
    """Check if `filename` is an image file."""  
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif'])  
  
def classify_images_by_grayscale(source_folder, target_bright_folder):  
    """  
    Classify images in `source_folder` by their grayscale testues.  
    Images with grayscale > 100 are saved to `target_bright_folder`.  
    Images with grayscale <= 100 are saved to `target_dark_folder`.  
    """  
    # 确保目标文件夹存在  
    os.makedirs(target_bright_folder, exist_ok=True)  
 
  
    # 遍历源文件夹中的所有文件  
    for filename in os.listdir(source_folder):  
        if is_image_file(filename):  
            img_path = os.path.join(source_folder, filename)  
            img = Image.open(img_path)   
            img_np = np.array(img)  # 将PIL图像转换为numpy数组 

            
            brightness = np.mean(img_np, axis=(0, 1))  # 这会给出R, G, B的平均值，但我们只想要一个值  
            # 但上面的代码实际上给出了每个颜色通道的平均值，我们取平均值来近似整体亮度  
            avg_brightness = np.mean(brightness) 
  
            # 根据灰度值分类图片  
            if avg_brightness > 100:  
                target_path = os.path.join(target_bright_folder, filename)    
                img.save(target_path)  # 保存图片到相应文件夹  
  
if __name__ == "__main__":  
    source_folder = '/home/mjy/ultralytics3/dataset/images/test'  
    target_bright_folder = '/home/mjy/ultralytics3/dataset/rgb/test'  

  
    classify_images_by_grayscale(source_folder, target_bright_folder)