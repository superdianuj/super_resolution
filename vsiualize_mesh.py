import open3d as o3d
import numpy as np
import imageio
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='Path to the  mesh file')
args = parser.parse_args()


# Load the mesh from a .ply file
mesh = o3d.io.read_triangle_mesh(args.path)

# Create a visualizer and add the mesh
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)  # Create a hidden window
vis.add_geometry(mesh)

# Set the camera parameters
ctr = vis.get_view_control()
ctr.set_zoom(0.8)
ctr.set_lookat([0, 0, 0])
ctr.set_up([0, 1, 0])

# Initialize the video writer
writer = imageio.get_writer(args.path+'_output.mp4', fps=30)

# Rotate the mesh and capture frames
for i in range(360):
    ctr.rotate(10.0, 0.0)
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(False)
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    writer.append_data(image)

# Close the video writer and the window
writer.close()
vis.destroy_window()