# What the driving policy sees of its target object: the YOLO box of the
# object and the depth at the box. Specified in docs/interfaces.md (M3).
# When visible is false, x, y, w, h, z are 0 and z_valid is false.

std_msgs/Header header   # stamp of the color image the observation is based on
bool visible             # target class detected this frame
float32 x                # box center, -1 left .. 1 right
float32 y                # box center, -1 top .. 1 bottom
float32 w                # box width / image width
float32 h                # box height / image height
bool z_valid             # depth reading between 0.2 and 1.0 m available
float32 z                # m, distance from the camera to the object surface; 0 if not valid
float32 score            # YOLO confidence, 0 .. 1
string class_id          # COCO class name, for example "sports ball"
