from visionserve import Client, VisionServeError, draw, select_target_grasp
import cv2
from PIL import Image

client = Client("http://10.252.205.103:11435")


rgb = cv2.imread("clients/python/examples/grasp_sample.jpg")[...,::-1]
model = "grasp-gd"
try:
    client.load(model)
    result = client.predict(
        model,
        rgb,
        # prompt='object',
        # max_grasps_per_object=3,
        gripper_min=10,
        gripper_max=300,
        min_size=1,
        max_size=15
    )
except VisionServeError as e:
    print("server error: %s" % e)

# annotated = draw(result, Image.fromarray(rgb))
target = select_target_grasp(result.grasps)
annotated = result.visualize(rgb, target_grasp=target)

annotated.save('out.png')

# result = client.predict('', rgb)


    
aa = 1