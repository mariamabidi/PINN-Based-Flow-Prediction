import pyvista as pv
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkBox
from vtkmodules.vtkFiltersExtraction import vtkExtractGeometry

# === Load mesh === #
filename = "/data/processed/updated.vtk"
mesh = pv.read(filename)

print("Available arrays:", mesh.array_names)  # Check for 'U' or similar

# === Crop around the car === #
xmin, xmax = -2.0, 5.0
ymin, ymax = -1.9, 0.0
zmin, zmax = 0.0, 2.5

box = vtkBox()
box.SetBounds(xmin, xmax, ymin, ymax, zmin, zmax)

extract = vtkExtractGeometry()
extract.SetImplicitFunction(box)
extract.SetInputData(mesh)
extract.Update()

cropped = pv.wrap(extract.GetOutput())

if cropped.n_points > 0:
    print("Cropped mesh has", cropped.n_points, "points.")

    # === Compute velocity magnitude === #
    velocity_vectors = cropped["U"]
    velocity_magnitude = np.linalg.norm(velocity_vectors, axis=1)
    cropped["velocity_magnitude"] = velocity_magnitude

    # === Define a plane to seed streamlines === #
    seed = pv.Plane(
        center=(0.5, -1.8, 1.0),    # Adjust if needed
        direction=(1, 0, 0),
        i_size=3.0,
        j_size=5.0,
        i_resolution=40,
        j_resolution=40
    )

    # === Generate streamlines using correct method === #
    streamlines = cropped.streamlines_from_source(
        seed,
        vectors="U",
        max_time=5.0,
        integrator_type=45,
        initial_step_length=0.1,
        terminal_speed=1e-12
    )

    # === Plot everything === #
    plotter = pv.Plotter()
    plotter.add_mesh(cropped, scalars="p", cmap="magma", opacity=0.4)
    plotter.add_mesh(streamlines.tube(radius=0.01), color="blue")
    plotter.add_mesh(seed, color="red", style="wireframe", opacity=0.8)
    plotter.show()

    cropped.save("car_cropped.vtk")

else:
    print("No points in cropped region.")