
import logging
import time
import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

mode=2
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    
    w, h = model_wh('432x368')
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path('cmu'), target_size=(w, h), trt_bool=str2bool('False'))
    else:
        e = TfPoseEstimator(get_graph_path('cmu'), target_size=(432, 368), trt_bool=str2bool('False'))
    #logger.debug('cam read+')
    cam = cv2.VideoCapture('path to video or external camera')
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    r=4.0
    count = 0
    y1 = [0,0]
    frame = 0
    while True:
        ret_val, image = cam.read()
        i =1
        count+=1
        if count % 11 == 0:
            continue
        # logger.debug('image process+')
        if not ret_val:
            break

        try:# logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=r)
    
            # logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=True)
    
            # logger.debug('show+')
            if mode == 1:
                hu = len(humans)
                print("Total no. of People : ", hu)
            elif mode == 2:
                for human in humans:
                    # we select one person from num of person
                    for i in range(len(humans)):
                        try:
                            a = human.body_parts.get(0)               
                            x = a.x*image.shape[1]   
                            y = a.y*image.shape[0]   
                            y1.append(y)  
                            print(a)
                        except:
                            pass
                            
                        if ((y - y1[-2]) > 36):  # it's distance between frame and comparing it with thresold value 
                            cv2.putText(image, "Fall Detected", (20,50), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0,0,255), 
                                2, 11)
                            print("fall detected.",i+1, count)#You can set count for get that your detection
            elif mode == 0:	
            	pass
        except:
            pass
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        out = cv2.VideoWriter('_output.mp4',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    20,(image.shape[1],image.shape[0]))
        out.write(image)
        if cv2.waitKey(1) == ord('q'):
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
