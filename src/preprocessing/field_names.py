import pyvista as pv

mesh = pv.read("car_cropped.vtk")
print(mesh.array_names)  # Lists all scalar/vector fields
