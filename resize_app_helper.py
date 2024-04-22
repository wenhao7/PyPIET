import numpy as np
import cv2
from ultralytics import YOLO
from IPython.display import Image, display
from PIL import ImageCms
import PIL
import os
import io

test_img_dir = 'images/test/test_convert/'

model = YOLO('models/best.pt')
person_model = YOLO('models/yolov8l.pt')

top_bottom_classes = list(range(13))



def detect(model, img_path, classes):
    """
    Model detects/predict objects in the image

    Parameters
    ----------
    model : ultralytics.models.yolo.model.YOLO
        Model chosen for the prediction task.
    img_path : str
        File path of image.
    classes : List
        List of class to predict depending on target classes the model has been trained on.

    Returns
    -------
    List
        List of ultralytics Results object containing information of the predicted objects.

    """
    return model(img_path, classes=classes)


def crop_image(img_path, product_res, h_pad=-1, w_pad=-1, person_res=None):
    """
    Crop image based on the bounding boxes identified by the model with padding if desired.
    Include additional ultralytics Result from a person detection model to refine object detection and prevent the target's face or shoes from being cropped.

    Parameters
    ----------
    img_path : str
        File path of image.
    product_res : List
        List of ultralytics Results object containing information of detected objects.
    h_pad : int, optional
        Number of pixels to pad top and bottom of crop. The default is -1.
    w_pad : int, optional
        Number of pixels to pad left and right of crop. The default is -1.
    person_res : TYPE, optional
        List of ultralytics Results object containing information of detected person. The default is None.

    Returns
    -------
    cropped_imgs : List
        List of numpy arrays containing the cropped images.
    paddings : List
        List of height and width padding applied to each of the cropped images.
    product_classes : List
        List of strings corresponding to each cropped image.

    """
    product_img_bgr = cv2.imread(img_path)
    product_img = cv2.cvtColor(product_img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = product_img.shape
    
    cropped_imgs = []
    paddings = []
    product_classes = []

    if h_pad == -1:
        h_pad = h//40
    if w_pad == -1:
        w_pad = w//40

    for i in range(len(product_res[0].boxes)):
        product_class = product_res[0].names[int(product_res[0].boxes.cls[i])]
        x1,y1,x2,y2 = product_res[0].boxes.xyxy[i].cpu().detach().numpy().astype(int)
        if person_res and len(person_res[0].boxes):  
            x3,y3,x4,y4 = person_res[0].boxes.xyxy[0].cpu().detach().numpy() .astype(int)
        else:
            x3,y3,x4,y4 = x1,y1,x2,y2
        if "top" in product_class:
            l = min( max(0, x1-w_pad), max(0, x3-w_pad))
            t = min( max(0, y1-h_pad), max(0, y3-h_pad))
            r = max( min(w, x2+w_pad), min(w, x4+w_pad))
            b = min(h, y2+h_pad)
        else:
            l = min( max(0, x1-w_pad), max(0, x3-w_pad))
            t = max(0, y1-h_pad)
            r = max( min(w, x2+w_pad), min(w, x4+w_pad))
            b = max( min(h, y2+h_pad), min(h, y4+h_pad))
        l, r, t, b = int(l), int(r), int(t), int(b)
        cropped_imgs.append(product_img[t:b, l:r])
        left_pad = x1-l
        right_pad = r-x2
        paddings.append((left_pad, right_pad))
        product_classes.append(product_class)
    return cropped_imgs, paddings, product_classes


def resize_pad(cropped_img, padding=None, target_res=(1270,1750), product_class='', crop_bottom = False):
    """
    Resize image to target resolution, padding to achieve aspect ratio if required.

    Parameters
    ----------
    cropped_img : numpy array
        Array of cropped img.
    padding : Tuple, optional
        Height and width padded when image was cropped. The default is None.
    target_res : Tuple, optional
        Desired height and width. The default is (1270,1750).
    product_class : str, optional
        Class of the cropped image. The default is ''.

    Returns
    -------
    numpy array
        Array of the final image.

    """
    #determine dimensions and directions to pad
    h_ratio, w_ratio = target_res[0]/int(cropped_img.shape[0]), target_res[1]/int(cropped_img.shape[1])
    extend_direction = h_ratio/w_ratio
    #print(h_ratio, w_ratio, extend_direction)
    #if ((extend_direction<1) & (h_ratio<w_ratio)) or ((extend_direction>=1) & (h_ratio>w_ratio)):
    target_h_img = target_res[0]
    ratio = target_h_img/ int(cropped_img.shape[0])
    target_w_img = int(ratio * int(cropped_img.shape[1]))
    if target_w_img > target_res[1]:
        ratio = target_w_img / target_res[1]
        target_w_img = int(int(target_w_img)/ratio)
        target_h_img = int(target_h_img/ratio)
    #print(target_h_img, target_w_img)
    img = PIL.Image.fromarray(cropped_img)
    #print("original img shape: ",np.array(img, np.uint8).shape)
    img = img.resize((target_w_img, target_h_img), PIL.Image.Resampling.LANCZOS)
    resized_image = np.array(img, np.uint8)
    #display(img)
    #print("resized img shape: ", resized_image.shape)
   
    resized_image = resized_image.transpose(2,0,1).reshape(resized_image.shape[2], resized_image.shape[0], resized_image.shape[1])
    img_arr = np.ndarray((3, resized_image.shape[1] + (target_res[0]-target_h_img), resized_image.shape[2] + (target_res[1]-target_w_img)), np.uint8)
    mask_arr = np.ndarray((3, resized_image.shape[1] + (target_res[0]-target_h_img), resized_image.shape[2] + (target_res[1]-target_w_img)), np.uint8)
    #print(f"resized/img_arr/mask_arr: {resized_image.shape}; {img_arr.shape}; {mask_arr.shape}")
    pad_l = int((target_res[1] - img.size[0])/2)
    pad_r = int(target_res[1] - pad_l - img.size[0])
    pad_t = int((target_res[0] - img.size[1])/2)
    pad_b = int(target_res[0] - pad_t - img.size[1])
    #print('Target res', target_res)
    #print('Img.size : ', img.size)
    #print(f"Padding pixels : {pad_l}, {pad_r}, {pad_t}, {pad_b}")
    
    #if "top" in product_class:
    #    pad_t = pad_t + pad_b
    #    pad_b = 0
    if crop_bottom and "bottom" in product_class:
        pad_b = pad_t + pad_b
        pad_t = 0
    #print(pad_l, pad_r, pad_t, pad_b)
    padded_l, padded_r = padding

    # extending edges of each colour channel separately
    for i, x in enumerate(resized_image):
        # obtain median color from padded width from cropping object detection, accounting for resizing ratio to avoid taking pixels from the product
        cons = (int(np.median(x[:,: int(padded_l*ratio)])), int(np.median(x[:, -int(padded_r*ratio):])))
        print("Cons: ", cons)
        
        # 2 part padding to stagger ramp up
        x_p = np.pad(x, ((pad_t//2,pad_b//2),(pad_l//2 , pad_r//2)), 'linear_ramp', end_values=cons)
        x_p = np.pad(x_p, ((pad_t-pad_t//2,pad_b-pad_b//2),(pad_l - pad_l//2 , pad_r - pad_r//2)), 'linear_ramp', end_values=cons)

        # mask starts from original cropped edge
        img_arr[i,:,:] = x_p
        #mask_arr[i,:,:pad_l] = 255
        #mask_arr[i,:,-pad_r:] = 255
        mask_arr[i,:pad_t, :pad_l] = 255
        mask_arr[i,-pad_b:, -pad_r:] = 255

    img_arr = np.uint8(img_arr).transpose(1,2,0)
    mask_arr = np.uint8(mask_arr).transpose(1,2,0)
    b = cv2.GaussianBlur(img_arr, (5,5), (0))
    img_arr = np.array(img_arr, np.uint8)
    # apply blur to mask
    img_arr[mask_arr==255] = b[mask_arr==255]
    #testimg = PIL.Image.fromarray(mask_arr)
    #testimg.save(str('./test'+str(target_res[1]) + 'x' + str(target_res[0]) + '.jpg'), quality=95, dpi=(300,300))
    img = PIL.Image.fromarray(img_arr)
    final_image = img.resize((target_res[1], target_res[0]), PIL.Image.Resampling.LANCZOS)
    #print("final img shape: ", np.array(final_image).shape)
    return np.array(final_image, np.uint8)


def resize_pad_white(cropped_img, padding=None, target_res=(1270,1750), product_class=''):
    """
    Resize image to target resolution, padding whitespace to achieve aspect ratio if required.
    For product images where no blending is required

    Parameters
    ----------
    cropped_img : numpy array
        Array of cropped img.
    padding : Tuple, optional
        Height and width padded when image was cropped. The default is None.
    target_res : Tuple, optional
        Desired height and width. The default is (1270,1750).
    product_class : str, optional
        Class of the cropped image. The default is ''.

    Returns
    -------
    numpy array
        Array of the final image.

    """
    #determine dimensions and directions to pad
    h_ratio, w_ratio = target_res[0]/int(cropped_img.shape[0]), target_res[1]/int(cropped_img.shape[1])
    extend_direction = h_ratio/w_ratio
    #print(h_ratio, w_ratio, extend_direction)
    target_h_img = target_res[0]
    ratio = target_h_img/ int(cropped_img.shape[0])
    target_w_img = int(ratio * int(cropped_img.shape[1]))
    if target_w_img > target_res[1]:
        ratio = target_w_img / target_res[1]
        target_w_img = int(int(target_w_img)/ratio)
        target_h_img = int(target_h_img/ratio)
    #print(target_h_img, target_w_img)
    img = PIL.Image.fromarray(cropped_img)
    #print("original img shape: ",np.array(img).shape)
    img = img.resize((target_w_img, target_h_img), PIL.Image.Resampling.LANCZOS)
    resized_image = np.array(img, np.uint8)
    #display(img)
    #print("resized img shape: ", resized_image.shape)
    
    resized_image = resized_image.transpose(2,0,1).reshape(resized_image.shape[2], resized_image.shape[0], resized_image.shape[1])
    img_arr = np.ndarray((3, resized_image.shape[1] + (target_res[0]-target_h_img), resized_image.shape[2] + (target_res[1]-target_w_img)), np.uint8)
    #print(f"resized/img_arr: {resized_image.shape}; {img_arr.shape}")
    pad_l = int((target_res[1] - img.size[0])/2)
    pad_r = int(target_res[1] - pad_l - img.size[0])
    pad_t = int((target_res[0] - img.size[1])/2)
    pad_b = int(target_res[0] - pad_t - img.size[1])
    
    for i, x in enumerate(resized_image):
        x_p = np.pad(x, ((pad_t,pad_b),(pad_l , pad_r)), 'constant', constant_values=(255,255))
        img_arr[i,:,:] = x_p
        
    img_arr = np.uint8(img_arr).transpose(1,2,0)
    img_arr = np.array(img_arr, np.uint8)
    img = PIL.Image.fromarray(img_arr)
    final_image = img.resize((target_res[1], target_res[0]), PIL.Image.Resampling.LANCZOS)
    #print("final img shape: ", np.array(final_image).shape)
    #display(img)
    return np.array(final_image, np.uint8)


def resize_no_detection(img_path, target_res):
    """
    Resize image and pad with whitespace without cropping or centering.

    Parameters
    ----------
    img_path : str
        File path to image.
    target_res : Tuple
        Target final resolution.

    Returns
    -------
    numpy array
        np array of final image.

    """
    product_img_bgr = cv2.imread(img_path)
    product_img = cv2.cvtColor(product_img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = product_img.shape

    h_ratio, w_ratio = target_res[0]/int(h), target_res[1]/int(w)
    extend_direction = h_ratio/w_ratio
    target_h_img = target_res[0]
    ratio = target_h_img/int(h)
    target_w_img = int(ratio * w)

    if target_w_img > target_res[1]:
        ratio = target_w_img / target_res[1]
        target_w_img = int(int(target_w_img)/ratio)
        target_h_img = int(target_h_img/ratio)
    img = PIL.Image.fromarray(product_img)
    img = img.resize((target_w_img, target_h_img), PIL.Image.Resampling.LANCZOS)
    resized_image = np.array(img, np.uint8)

    pad_l = int((target_res[1] - img.size[0])/2)
    pad_r = int(target_res[1] - pad_l - img.size[0])
    pad_t = int((target_res[0] - img.size[1])/2)
    pad_b = int(target_res[0] - pad_t - img.size[1])

    resized_image = resized_image.transpose(2,0,1).reshape(resized_image.shape[2], resized_image.shape[0], resized_image.shape[1])
    img_arr = np.ndarray((3, resized_image.shape[1] + (target_res[0]-target_h_img), resized_image.shape[2] + (target_res[1]-target_w_img)), np.uint8)

    for i, x in enumerate(resized_image):
        x_p = np.pad(x, ((pad_t, pad_b), (pad_l, pad_r)), 'constant', constant_values=255)
        img_arr[i,:,:] = x_p

    img_arr = np.uint8(img_arr).transpose(1,2,0)
    img_arr = np.array(img_arr, np.uint8)
    img = PIL.Image.fromarray(img_arr)
    final_image = img.resize((target_res[1], target_res[0]), PIL.Image.Resampling.LANCZOS)
    return np.array(final_image, np.uint8)


def get_filepaths(folder_path):
    """
    Get file path of images within the folder

    Parameters
    ----------
    folder_path : str
        Folder path directory.

    Returns
    -------
    file_paths : List
        List of file path to the images.

    """
    file_names = os.listdir(folder_path)
    file_paths = []
    for file_name in file_names:
        if file_name.endswith(('.tif','.jpg','.png','.jpeg')):
            file_paths.append(folder_path + file_name)
    return file_paths


def process_filename(file_path, image_format='jpg'):
    """
    Process new image file name

    Parameters
    ----------
    file_path : str
        Image file path.
    image_format : str, optional
        Output image format extension. The default is 'jpg'.

    Returns
    -------
    new_name : str
        New image file name.

    """
    img_name = file_path.split('/')[-1].split('_')
    img_name[4] = 'EAS'
    img_name[5] = 'PS'
    img_name[7] = ''.join(img_name[7].split('.')[0])
    new_name = '_'.join(img_name[:8]) + '.' + image_format
    return new_name


def save_to_dir(img_array, img_name, target_res, folder_name, im_profile=None):
    """
    Save image to drive

    Parameters
    ----------
    img_array : numpy array
        numpy array of image.
    img_name : str
        New image file name.
    target_res : Tuple
        Target resolution of image.
    folder_name : str
        File path of folder image will be saved in.

    Returns
    -------
    None.

    """
    new_path = str(folder_name + '/' + str(target_res[1]) + ' x ' + str(target_res[0]))
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    img = PIL.Image.fromarray(img_array)
    img.save(str(new_path + '/' + img_name), quality=95, dpi=(300,300), icc_profile=im_profile.tobytes())
    
def process_images_path(file_paths):
    """
    Preprocess folder paths and log file

    Parameters
    ----------
    file_paths : str
        Folder directory.

    Returns
    -------
    folder_name : TYPE
        DESCRIPTION.

    """
    folder_name = '/'.join(file_paths[0].split('/')[:-1]) + '/'
    try:
        os.remove(folder_name + 'check_images.txt')
    except OSError:
        pass
    
    return folder_name    

def process_image(file, target_res_dict, folder_name):
    """
    Sequentially executes other user defined functions to process original image file to final edits

    Parameters
    ----------
    file_paths : List
        List of file paths of images to edit.
    target_res_dict : Dict
        Dictionary containing folder names:target resolutions as key:value pairs.
    folder_name : str
        File path of folder directory.

    Returns
    -------
    None.

    """
   
    img_name = file.split('/')[-1]
    try:
        img_new_name = process_filename(file)
    except IndexError:
        print(f' !!!!!!!!!!!!!!! File Name {img_name} does not meet convention !!!!!!!!!!!!!!! ')
        with open(folder_name + 'check_images.txt', 'a') as f:
            f.write(img_name + '\n')
            f.write("Bad file name" + '\n')
    except Exception as e:
        s = "Error {0}".format(str(e))       
        with open(folder_name + 'check_images.txt', 'a') as f:
            f.write(img_name + '\n')
            f.write(s + '\n')
    
    person_res = detect(person_model, file, classes=0)
    person_cropped, _, _ = crop_image(file, person_res)
    
    product_res = detect(model, file, classes=top_bottom_classes)
    product_cropped, paddings, product_classes = crop_image(file, product_res, person_res=person_res)
    bottom_log, bottom_done = 0, 0
    
    is_bottom = 'WB' in img_name or 'MB' in img_name
    is_laydown_detail = 'LD_D1' in img_name or 'LD_A1' in img_name
    is_on_detail = 'ON_D1' in img_name or 'ON_A1' in img_name
    
    im = PIL.Image.open(file)
    im_profile = ImageCms.ImageCmsProfile(io.BytesIO(im.info.get('icc_profile')))

    try:
        for i in range(len(product_cropped)):
            if (('WB' in img_name or 'MB' in img_name) and 'ON_FV' in img_name) and 'bottom' not in product_classes[i]:
                bottom_log = 1
            for k, v in target_res_dict.items():
                for target_res in v:
                    if (is_bottom and 'ON_FV' in img_name) and 'bottom' in product_classes[i] and not is_laydown_detail and not is_on_detail:
                        final_product = resize_pad(product_cropped[i], paddings[i], target_res=target_res, product_class=product_classes[i], crop_bottom=True)
                        #img = PIL.Image.fromarray(final_product)

                        img_format = img_new_name.split('.')[-1]
                        img_ON_FV_D2 = img_new_name.split('_')
                        img_ON_FV_D2[-1] = 'D2.' + str(img_format)
                        img_ON_FV_D2 = '_'.join(img_ON_FV_D2)
                        save_to_dir(final_product, img_ON_FV_D2, target_res, folder_name + k, im_profile=im_profile)
                        #print('A', img.size)
                        #print(k, target_res)
                        #display(img)
                        bottom_done = 1

                    elif 'LD' in img_name and not is_laydown_detail:
                        final_product = resize_pad(product_cropped[i], paddings[i], target_res=target_res, product_class=product_classes[i])
                        #img = PIL.Image.fromarray(final_product)
                        save_to_dir(final_product, img_new_name, target_res, folder_name + k, im_profile=im_profile)
                        #print('C', img.size)
                        #print(k, target_res)
                        #display(img)

                    if (i==0 and person_cropped and not is_laydown_detail and not is_on_detail) :
                         #print('=============Person================')
                        final_person = resize_pad(person_cropped[0], paddings[0], target_res=target_res)
                        #img_person = PIL.Image.fromarray(final_person)
                        save_to_dir(final_person, img_new_name, target_res, folder_name + k, im_profile=im_profile)
                        #print('B', img_person.size)
                        #print(k, target_res)
                        #display(img_person)

                    if is_laydown_detail : #or is_on_detail:
                        final_product = resize_pad_white(product_cropped[i], paddings[i], target_res=target_res, product_class=product_classes[i])
                        #img = PIL.Image.fromarray(final_product)

                        img_format = img_new_name.split('.')[-1]
                        img_LD_D1 = img_new_name.split('_')
                        img_LD_D1[-1] = 'D1.' + str(img_format)
                        img_LD_D1 = '_'.join(img_LD_D1)
                        save_to_dir(final_product, img_LD_D1, target_res, folder_name + k, im_profile=im_profile)
                        #print('D', img.size)
                        #print(k, target_res)
                        #display(img)
                        
                    elif is_on_detail:
                        if person_cropped:
                            print('detail person padding white')
                            final_product = resize_pad_white(person_cropped[i], paddings[i], target_res=target_res, product_class=product_classes[i])
                        else:
                            final_product = resize_no_detection(file, target_res=target_res)
                        #img = PIL.Image.fromarray(final_product)
                        img_format = img_new_name.split('.')[-1]
                        img_LD_D1 = img_new_name.split('_')
                        img_LD_D1[-1] = 'D1.' + str(img_format)
                        img_LD_D1 = '_'.join(img_LD_D1)
                        save_to_dir(final_product, img_LD_D1, target_res, folder_name + k, im_profile=im_profile)
                        #print('F', img.size)
                        #print(k, target_res)
                        #display(img)
                    
                    if is_laydown_detail or is_on_detail or 'LD' in img_name:
                        alt_product = resize_no_detection(file, target_res = target_res)
                        #img = PIL.Image.fromarray(alt_product)
                        img_format = img_new_name.split('.')[-1]
                        alt_name = img_new_name.split('_')
                        alt_name[-1] = alt_name[-1] + '_ALT.' + str(img_format)
                        alt_name = '_'.join(alt_name)
                        save_to_dir(alt_product, alt_name, target_res, folder_name + k, im_profile=im_profile)
                        #print('G alt', img.size)
                        #print(k, target_res)
                        #display(img)
                        
        if len(product_cropped) == 0:
            with open(folder_name + 'check_images.txt', 'a') as f:
                f.write(img_name + '\n')
                f.write('No object detected, image may have been padded without centering or skipped completely. \n')
        if (len(product_cropped) == 0): # and ('LD_D1' in img_name or 'LD_A1' in img_name):
            for k, v in target_res_dict.items():
                for target_res in v:
                    final_product = resize_no_detection(file, target_res)
                    img = PIL.Image.fromarray(final_product)

                    if is_laydown_detail:
                        img_format = img_new_name.split('.')[-1]
                        img_LD_D1 = img_new_name.split('_')
                        img_LD_D1[-1] = 'D1.' + str(img_format)
                        img_new_name = '_'.join(img_LD_D1)

                    save_to_dir(final_product, img_new_name, target_res, folder_name + k, im_profile=im_profile)
                    #print('E', img.size)
                    #print(k, target_res)
                    #display(img)
        if bottom_log and not bottom_done:
            with open(folder_name + 'check_images.txt', 'a') as f:
                f.write(img_name + '\n')
                f.write('No bottoms detected for front view bottom image, cropped front bottom view is skipped for this file \n')
            bottom_log = 0
    except Exception as e:
        s = "Error {0}".format(str(e))
        r = ''
        if target_res:
            r = str(target_res[1]) + 'x' + str(target_res[0])
        with open(folder_name + 'check_images.txt', 'a') as f:
            f.write(img_name + '    ' + r + '\n')
            f.write(s + '\n')