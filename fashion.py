#!/usr/bin/env python3

import argparse
import numpy as np
import sys
import cv2

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

import time

# Classifier input size.
INPUT_HEIGHT = 224
INPUT_WIDTH = 224

def preprocess(image):
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    # image = image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.ANTIALIAS)
    # image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(np.array(image, dtype=np.float32, order='C'), (2, 0, 1))

    # Inception.
    image /= 127.5
    image -= 1
    
    # Yolov3/V4.
    # image /= 255.0

    return image

def fashion(path):
    # Create server context.
    try:
        triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001',
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)
    except Exception as e:
        print("Context creation failed: " + str(e))
        sys.exit()

    # Health check.
    if not triton_client.is_server_live():
        print("FAILED: is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED: is_server_ready")
        sys.exit(1)
    
    if not triton_client.is_model_ready("fashion"):
        print("FAILED: is_model_ready")
        sys.exit(1)
    # To display the meta.
    # python3 client.py image fashion.jpg -i
    # if FLAGS.model_info:
    #     # Model metadata.
    #     try:
    #         metadata = triton_client.get_model_metadata("fashion")
    #         print(metadata)
    #     except InferenceServerException as ex:
    #         if "Request for unknown model" not in ex.message():
    #             print("FAILED : get_model_metadata")
    #             print("Got: {}".format(ex.message()))
    #             sys.exit(1)
    #         else:
    #             print("FAILED : get_model_metadata")
    #             sys.exit(1)

    #     # Model configuration.
    #     try:
    #         config = triton_client.get_model_config("fashion")
    #         if not (config.config.name == FLAGS.model):
    #             print("FAILED: get_model_config")
    #             sys.exit(1)
    #         print(config)
    #     except InferenceServerException as ex:
    #         print("FAILED : get_model_config")
    #         print("Got: {}".format(ex.message()))
    #         sys.exit(1)

    
    inputs = []
    outputs = []
    # Specify the input size for classifier.
    inputs.append(grpcclient.InferInput('input_1', [1, 3, INPUT_HEIGHT, INPUT_WIDTH], "FP32"))
    # Specify the output nodes.
    outputs.append(grpcclient.InferRequestedOutput('output_node0'))
    outputs.append(grpcclient.InferRequestedOutput('output_node1'))
    outputs.append(grpcclient.InferRequestedOutput('output_node2'))
    outputs.append(grpcclient.InferRequestedOutput('output_node3'))
    outputs.append(grpcclient.InferRequestedOutput('output_node4'))
    outputs.append(grpcclient.InferRequestedOutput('output_node5'))
    outputs.append(grpcclient.InferRequestedOutput('output_node6'))
    outputs.append(grpcclient.InferRequestedOutput('output_node7'))

    input_image = cv2.imread(path)
    # input_image = Image.open(str(FLAGS.input)) # .convert('RGB')

    if input_image is None:
        return None # Need to check out of the function.
    input_image_buffer = preprocess(input_image)

    # For the shape.
    # input_image = np.array(input_image.convert('RGB'))

    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    inputs[0].set_data_from_numpy(input_image_buffer)

    results = triton_client.infer(model_name="fashion",
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=None)
    # if FLAGS.model_info:
    #     statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
    #     if len(statistics.model_stats) != 1:
    #         print("FAILED: get_inference_statistics")
    #         sys.exit(1)
    #     print(statistics)
    
    outputs_res = [results.as_numpy('output_node0'), # neck_design.
                   results.as_numpy('output_node1'), # coat_length.
                   results.as_numpy('output_node2'), # sleeve_length.
                   results.as_numpy('output_node3'), # neckline_design
                   results.as_numpy('output_node4'), # collar_design.
                   results.as_numpy('output_node5'), # pant_length.
                   results.as_numpy('output_node6'), # skirt_length.
                   results.as_numpy('output_node7')] # lapel_design.

    labels_path = ["labels/neck_design.txt", "labels/coat_length.txt",
                   "labels/sleeve_length.txt", "labels/neckline_design.txt",
                   "labels/collar_design.txt", "labels/pant_length.txt",
                   "labels/skirt_length.txt", "labels/lapel_design.txt"]
    all_labels = []

    for label_file in labels_path:
        f = open(label_file, 'r')
        l = f.readlines()
        f.close()
        labels = []
        for t in l:
            labels.append(t.split()[0])
        all_labels.append(labels)
    
    classifier_output = len(outputs_res)

    res = {
        "timeUsed": 0.063, "predictions": {
            "image_0": {
                "neck_design": [],
                "coat_length": [],
                "sleeve_length": [],
                "neckline_design": [],
                "collar_design": [],
                "pant_length": [],
                "skirt_length": [],
                "lapel_design": []
            }
        }, "success": True
    }

    mapping = {0: "neck_design",
               1: "coat_length",
               2: "sleeve_length",
               3: "neckline_design",
               4: "collar_design",
               5: "pant_length",
               6: "skirt_length",
               7: "lapel_design"}
    for i in range(classifier_output):
        output = outputs_res[i].reshape(-1,) # For top-k, [0.02321227 0.03176818 0.00130235 0.05652939 0.8871879].
        idx = output.argsort()[-3:][::-1] # [4 3 1].
        # print(all_labels[i], idx, output[0][idx], all_labels[i][idx])

        for j in range(len(idx)):
            res["predictions"]["image_0"][mapping[i]].append({"confidence": float(output[idx[j]]), "label": all_labels[i][idx[j]]})
        
    return res
    """
['Invisible', 'TurtleNeck', 'RuffleSemi-HighCollar', 'LowTurtleNeck', 'DrapedCollar'] 4 [[0.8871879]] DrapedCollar
['Invisible', 'HighWaist', 'Regular', 'Long', 'Micro', 'Knee', 'Midi', 'AnkleFloor'] 7 [[0.99978024]] AnkleFloor
['Invisible', 'Sleeveless', 'CupSleeves', 'ShortSleeves', 'ElbowSleeves', '3/4Sleeves', 'WristLength', 'LongSleeves', 'ExtraLongSleeves'] 8 [[0.6724836]] ExtraLongSleeves
['Invisible', 'Strapless', 'DeepVNeckline', 'Straight', 'VNeckline', 'SquareNeckline', 'OffShoulder', 'RoundNeckline', 'SweatHeartNeck', 'OneShoulderNeckline'] 0 [[0.29348585]] Invisible
['Invisible', 'Shirt', 'Peter', 'Puritan', 'Rib'] 0 [[0.91322947]] Invisible
['Invisible', 'ShortPant', 'MidLength', '3/4Length', 'CroppedPant', 'FullLength'] 0 [[0.9781059]] Invisible
['Invisible', 'Short', 'Knee', 'Midi', 'Ankle', 'Floor'] 5 [[0.99832827]] Floor
['Invisible', 'Notched', 'Collarless', 'ShawlCollar', 'PlusSizeShawl'] 0 [[0.99026287]] Invisible
    """
    

    # detected_objects = postprocess(result, input_image.shape[1], input_image.shape[0], FLAGS.confidence, FLAGS.nms)
    # print(f"Raw boxes: {int(result[0, 0, 0, 0])}")
    # print(f"Detected objects: {len(detected_objects)}")

    # for box in detected_objects:
    #     # print(f"{COCOLabels(box.classID).name}: {box.confidence}")
    #     input_image = render_box(input_image, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
    #     size = get_text_size(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", normalised_scaling=0.6)
    #     input_image = render_filled_box(input_image, (box.x1-3, box.y1-3, box.x1+size[0], box.y1+size[1]), color=(220, 220, 220))
    #     input_image = render_text(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)

    # if FLAGS.out:
    #     cv2.imwrite(FLAGS.out, input_image)
    #     print(f"Saved result to {FLAGS.out}")
    # else:
    #     cv2.imshow('image', input_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

