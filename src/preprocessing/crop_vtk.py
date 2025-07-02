import pyvista as pv
from vtkmodules.vtkCommonDataModel import vtkBox
from vtkmodules.vtkFiltersExtraction import vtkExtractGeometry

# Load VTK file
filename = "../data/raw/E_S_WWC_WM_005.vtk"  # replace with your actual file name
mesh = pv.read(filename)

# Plot full mesh to estimate car region
print("Displaying full mesh. Use this to determine the car's bounding box.")
mesh.plot(scalars="U")
# Crop around estimated car region (adjust bounds after visual inspection)
xmin, xmax = -4.0, 10.0
ymin, ymax = -1.9, 0.0
zmin, zmax = -3.469446951953614e-18, 2.5

# Create VTK box
box = vtkBox()
box.SetBounds(xmin, xmax, ymin, ymax, zmin, zmax)

# Extract geometry
extract = vtkExtractGeometry()
extract.SetImplicitFunction(box)
extract.SetInputData(mesh)
extract.Update()

# Convert to PyVista mesh
cropped = pv.wrap(extract.GetOutput())

# Plot and save
if cropped.n_points > 0:
    print("✅ Cropped mesh has", cropped.n_points, "points.")
    cropped.plot(scalars="U")
    cropped.save("car_cropped.vtk")
else:
    print("❌ No points in cropped region.")
