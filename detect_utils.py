import torchvision.transforms as transforms
import cv2
import numpy
import numpy as np

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

#create different color for each class
COLORS = np.random.uniform(0,255,size = (len(coco_names),3))

#define transforms
transform = transforms.Compose([transforms.ToTensor(),])

def predict(image,model,device,detection_threshold):
    # transfrom the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0)  #add a batch dimension
    outputs = model(image)  #get the predictions on the image

    #get all the prediction class names
    pred_classes = [coco_names[i] for i in outputs[0]['label'].cpu().numpy()]

    #get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    #get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    #get boxes above the threshold score
    boxes = pred_bboxes[pred_scores>=detection_threshold].astype(np.int32)

    return boxes,pred_classes,outputs[0]['label']

#draw bounding boxes around an object
def draw_boxes(boxes,classes,labels,image):
    #read the image with the opencv
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_BGR2RGB)
    for i , box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]),int(box[1])),
            (int(box[2]),int(box[3])),
            color,2
        )
        cv2.putText(image,classes[i],(int(box[0]),int(box[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2,lineType=cv2.LINE_AA)
    return image



