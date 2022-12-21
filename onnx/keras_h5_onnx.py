import os

os.system('python3 -m tf2onnx.convert --saved-model ../cnn/face_net/facenet_nyoki --output ./facenet.onnx '
          '--opset=10')
