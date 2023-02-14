import os

# os.system('python3 -m tf2onnx.convert --saved-model ../cnn/face_net/face_net_nyoki --output ./facenet.onnx '
#           '--opset=10')
os.system('python -m tf2onnx.convert --saved-model ../cnn/face_net/face_net_nyoki --output ./facenet.onnx --opset 13')
