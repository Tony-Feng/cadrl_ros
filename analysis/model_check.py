# https://blog.csdn.net/c20081052/article/details/82961988
# https://blog.csdn.net/qq_28808697/article/details/79884309
from tensorflow.python import pywrap_tensorflow

##checkpoint_path = "network_02360000"
checkpoint_path = "network_01900000"
##checkpoint_path = "network_01653000"
model = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = model.get_variable_to_shape_map()
for i, j in var_to_shape_map.items():
    print(i, j)
