import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import colorsys
from scipy.spatial import ConvexHull, QhullError
from screeninfo import get_monitors
import argparse

from core.geometry import create_line_set_from_mesh
from core.planes import plane_ransac_mesh, plane_ransac_points, remove_planes_mesh, remove_planes_points
from core.clustering import dbscan
from core.grid import create_occupancy_grid, expand_obstacles
from core.path_planning import a_star_search

from config import PANEL_WIDTH, CAMERA_FOV_DEG
from config import PCD_NUM_POINTS, PCD_BASE_COLOR, MESH_BASE_COLOR
from config import CLUSTERING_EPS, CLUSTERING_MIN_SAMPLES
from config import MIN_DOOR_WIDTH, MAX_DOOR_WIDTH, MIN_DOOR_HEIGHT, Z_THRESHOLD, DOOR_CANDIDATE_COLOR
from config import GRID_RESOLUTION, DEFAULT_OBJECT_RADIUS, MAX_TEX
from config import PATH_POINT_SIZE, MARKER_SIZE, PATH_THICKNESS, PATH_COLOR, START_COLOR, GOAL_COLOR

class MeshApp:
    """
    Open3D GUI application for indoor path planning.
    
    Contains: 
        - an Open3D window and a 3D scene widget 
        - UI panel controls (buttons/slider)
        - 2D occupancy grid widget for start/goal points
    """
    def __init__(self, mesh, line_set, screen_w, screen_h):
        self.mesh = mesh
        self.line_set = line_set
        self.lines_shown = True
        self._camera_initialized = False
        self._path_geom_name = "path"
        self.pcd_flag = False          
        self.pcd = None         
        
        self.found_planes = []   
        self.current_inliers = None  
        
        self.labels = None
        self.door_hull_points = None
        self.door_label = None
                
        self.x_min = None
        self.y_min = None
        self.x_max = None
        self.y_max = None
              
        self.start_point_grid = None
        self.goal_point_grid = None
        self.start_point_world = None
        self.goal_point_world = None
        self._start_px = None       
        self._goal_px = None
        self.path = None
        
        self.grid_resolution = GRID_RESOLUTION
        self._grid_step = 1    
        self.radius = float(DEFAULT_OBJECT_RADIUS)
        self._grid_base_img = None  
        self.occupancy_grid = None
        self.expanded_grid = None
        self.grid_active = False

        # --- window ---
        self.window = gui.Application.instance.create_window(
            "Open3D GUI",
            screen_w - 50,
            screen_h - 50
        )

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        
        self.panel = gui.Vert(8, gui.Margins(10, 10, 10, 10))
        
        self.status = gui.Label("Ready.")

        # --- mesh settings ---
        self.mesh_mat = rendering.MaterialRecord()
        self.mesh_mat.shader = "defaultLitTransparency"  
        self.mesh_mat.base_color = [0.75, 0.75, 0.75, 0.75] 

        # --- line settings ---
        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 1.0
        
        # --- point cloud settings ---
        self.pcd_mat = rendering.MaterialRecord()
        self.pcd_mat.shader = "defaultUnlit"
        self.pcd_mat.point_size = 4.0

        # --- add geometries ---
        self.scene.scene.add_geometry("mesh", self.mesh, self.mesh_mat)
        self.scene.scene.add_geometry("wire", self.line_set, self.line_mat)

        # --- right panel (controls) ---
        self.btn_toggle_lines = gui.Button("Toggle wireframe")
        self.btn_toggle_lines.set_on_clicked(self.on_toggle_lines)

        self.panel.add_child(self.btn_toggle_lines)
        self.panel.add_child(self.status)

        self.window.add_child(self.scene)
        self.window.add_child(self.panel)

        self.window.set_on_layout(self._on_layout)
    
        # RANSAC buttons
        self.btn_next_plane = gui.Button("Select plane")
        self.btn_remove_plane = gui.Button("Remove highlighted plane")
        self.btn_recompute = gui.Button("RESET RANSAC")

        self.btn_next_plane.set_on_clicked(self.on_next_plane)
        self.btn_remove_plane.set_on_clicked(self.on_remove_plane)
        self.btn_recompute.set_on_clicked(self.on_recompute_planes)

        self.panel.add_child(self.btn_next_plane)
        self.panel.add_child(self.btn_remove_plane)
        self.panel.add_child(self.btn_recompute)
        
        self.btn_sample_or_back  = gui.Button("Sample mesh/ Create PointCloud")
        
        self.btn_sample_or_back.set_on_clicked(self.on_sample_or_back)
        
        self.panel.add_child(self.btn_sample_or_back)

        self.btn_cluster = gui.Button("Cluster (DBSCAN)")
        self.btn_cluster.set_on_clicked(self.on_cluster)
        self.panel.add_child(self.btn_cluster)

        self.btn_find_door = gui.Button("Find door candidates")
        self.btn_find_door.set_on_clicked(self.on_find_door_candidates)
        self.panel.add_child(self.btn_find_door)
        
        self.btn_grid = gui.Button("Create 2D Projection Grid")
        self.panel.add_child(self.btn_grid)

        self.grid_img = gui.ImageWidget()
        self.grid_img.background_color = gui.Color(1, 1, 1)
        self.panel.add_child(self.grid_img)
        
        placeholder = np.ones((64, 64, 3), dtype=np.uint8) * 255
        self.grid_img.update_image(o3d.geometry.Image(np.ascontiguousarray(placeholder)))

        self.btn_grid.set_on_clicked(self.on_grid_toggle)
        self.grid_img.set_on_mouse(self._ignore_mouse) 

        self.radius_label = gui.Label(f"Radius: {self.radius:.2f} m")
        self.radius_slider = gui.Slider(gui.Slider.DOUBLE)
        self.radius_slider.set_limits(0.0, 1.0)        
        self.radius_slider.double_value = self.radius
        self.radius_slider.set_on_value_changed(self.on_radius_changed)

        self.panel.add_child(self.radius_label)
        self.panel.add_child(self.radius_slider)
        
        self.btn_path = gui.Button("Find Path")
        self.btn_path.set_on_clicked(self.on_find_path)
        self.panel.add_child(self.btn_path)
        
    def _on_layout(self, layout_context):
        """
        Position the 3D scene and right-side control panel; initiliaze camera once.
        """
        r = self.window.content_rect
        panel_width = PANEL_WIDTH

        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.x + r.width - panel_width, r.y, panel_width, r.height)

        if not self._camera_initialized:
            bounds = self.mesh.get_axis_aligned_bounding_box()
            self.scene.setup_camera(CAMERA_FOV_DEG, bounds, bounds.get_center())
            self._camera_initialized = True
            
    # --- Scene and geometry helpers (3D)
    def _replace_mesh_and_wire(self):
        """
        Refresh the scene after mesh changes. 
        """
        try: 
            self.scene.scene.remove_geometry("mesh")
        except Exception:
            pass
        self.scene.scene.add_geometry("mesh", self.mesh, self.mesh_mat)

        try:
            self.scene.scene.remove_geometry("wire")
        except Exception:
            pass
        
        if self.lines_shown:            
            self.scene.scene.add_geometry("wire", self.line_set, self.line_mat)
   
    def _replace_pcd(self):
        """
        Refresh the point cloud in the scene.
        """
        try:
            self.scene.scene.remove_geometry("pcd")
        except Exception:
            pass
        self.scene.scene.add_geometry("pcd", self.pcd, self.pcd_mat)

    def on_toggle_lines(self):
        """
        Toggle the mesh wireframe overlay on/off.
        """
        self.lines_shown = not self.lines_shown

        if self.lines_shown:
            try:
                self.scene.scene.remove_geometry("wire")
            except Exception:
                pass
            self.scene.scene.add_geometry("wire", self.line_set, self.line_mat)
            self.status.text = "Wireframe ON"
        else:
            try:
                self.scene.scene.remove_geometry("wire")
            except Exception:
                pass
            self.status.text = "Wireframe OFF"
            
    def _reset_after_pcd_change(self):
        """Call whenever self.pcd.points changes (resample, remove planes/points, filtering, etc)."""
        self.labels = None
        self.occupancy_grid = None
        self.expanded_grid = None

        self.start_point_grid = None
        self.goal_point_grid = None
        self.start_point_world = None
        self.goal_point_world = None

        self.door_hull_points = None
        self.door_label = None

        self.found_planes = []
        self.current_inliers = None
        
    def _reset_after_labels_change(self):
        """Call whenever clustering labels are recomputed."""
        self.occupancy_grid = None
        self.expanded_grid = None

        self.start_point_grid = None
        self.goal_point_grid = None
        self.start_point_world = None
        self.goal_point_world = None

    def _reset_after_grid_change(self):
        """Call whenever occupancy/expanded grid is recomputed."""
        self.start_point_grid = None
        self.goal_point_grid = None
        self.start_point_world = None
        self.goal_point_world = None
        
    # --- Mesh/ Point cloud workflow
    def on_sample_pcd(self):
        """
        Sample points on the mesh and switch the 3D view to point-cloud mode.
        """
        pcd_num_points = PCD_NUM_POINTS

        self.found_planes = []
        self.current_inliers = None

        triangles = np.asarray(self.mesh.triangles, dtype=np.int64)
        vertices  = np.asarray(self.mesh.vertices, dtype=np.float64)

        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        total_area = tri_areas.sum()
        if total_area <= 0:
            self.status.text = "Mesh area is zero; cannot sample."
            return

        pts_per_tri = np.ceil((tri_areas / total_area) * pcd_num_points).astype(int)

        sampled = []
        for tri, k in zip(triangles, pts_per_tri):
            if k <= 0:
                continue
            A, B, C = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

            r1 = np.random.rand(k)
            r2 = np.random.rand(k)
            sqrt_r1 = np.sqrt(r1)

            alpha = 1.0 - sqrt_r1
            beta  = sqrt_r1 * (1.0 - r2)
            gamma = sqrt_r1 * r2

            pts = alpha[:, None] * A + beta[:, None] * B + gamma[:, None] * C
            sampled.append(pts)

        if not sampled:
            self.status.text = "No points sampled."
            return

        sampled_points = np.vstack(sampled)

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(sampled_points)
        self.pcd.colors = o3d.utility.Vector3dVector(np.tile(PCD_BASE_COLOR, (len(sampled_points), 1)))

        for name in ("mesh", "wire", "pcd"):
            try:
                self.scene.scene.remove_geometry(name)
            except Exception:
                pass

        self.scene.scene.add_geometry("pcd", self.pcd, self.pcd_mat)

        self.pcd_flag = True
        self.btn_sample_or_back.text = "Back to mesh"
        self.status.text = "Point cloud view."

        self._reset_after_pcd_change()

    def on_back_to_mesh(self):
        """
        Switch from point-cloud mode back to mesh mode in the 3D scene.
        """
        if not self.pcd_flag:
            return

        #TODO: fix mesh if points are removed 
        
        self.current_inliers = None
        self.found_planes = []

        for name in ("pcd", self._path_geom_name, "mesh", "wire"):
            try:
                self.scene.scene.remove_geometry(name)
            except Exception:
                pass

        self.scene.scene.add_geometry("mesh", self.mesh, self.mesh_mat)
        if self.lines_shown:
            self.scene.scene.add_geometry("wire", self.line_set, self.line_mat)
  
        self.pcd_flag = False
        self.btn_sample_or_back.text = "Sample mesh/ Create PointCloud"
        self.status.text = "Mesh view."
        
    def on_sample_or_back(self):
        """ 
        Toggle between mesh and point-cloud view.
        """
        if self.pcd_flag and self.pcd is not None:
            try:
                self.scene.scene.remove_geometry(self._path_geom_name)
            except Exception:
                pass
            self.on_back_to_mesh()
        else:
            self.on_sample_pcd()
            
    # --- Plane detection/ removal (RANSAC)
    def on_next_plane(self):
        """
        Highlight the next detected plane by coloring its inliers red.
        
        - Mesh mode: RANSAC on the mesh vertices 
        - Point-cloud mode: RANSAC on the sampled point-cloud.
        """
                    
        if self.pcd_flag and self.pcd is not None:
            colors = np.tile(PCD_BASE_COLOR, (len(self.pcd.points), 1))
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            if not self.found_planes:
                planes = plane_ransac_points(self.pcd)
                self.found_planes = [(p[1], p[3]) for p in planes]

            if not self.found_planes:
                self.current_inliers = None
                self.status.text = "No planes found (PCD)."
                self._replace_pcd()
                return

            _, inliers = self.found_planes.pop(0)
            self.current_inliers = np.asarray(inliers, dtype=np.int64)

            colors = np.asarray(self.pcd.colors)
            colors[self.current_inliers, :] = [1, 0, 0]
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            self.status.text = f"Highlighted plane (PCD) with {len(self.current_inliers)} inliers."
            self._replace_pcd()
            return
    
        colors = np.tile(MESH_BASE_COLOR, (len(self.mesh.vertices), 1))
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        if not self.found_planes:
            planes = plane_ransac_mesh(self.mesh)  
            self.found_planes = [(p[1], p[3]) for p in planes]
            if not self.found_planes:
                self.status.text = "No planes found."
                self._replace_mesh_and_wire()
                return

        _, inliers = self.found_planes.pop(0)
        self.current_inliers = np.asarray(inliers, dtype=np.int64)

        colors = np.asarray(self.mesh.vertex_colors)
        colors[self.current_inliers, :] = [1, 0, 0]
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        self.status.text = f"Highlighted plane with {len(self.current_inliers)} inliers."
        self._replace_mesh_and_wire()
        
    def on_remove_plane(self):
        """
        Remove the highlighted plane, reset plane cache and refresh the scene.
        """
        if self.path is not None:
            self._update_grid_image()
            self.scene.scene.remove_geometry(self._path_geom_name)
        if self.grid_active:
            self.btn_grid.text = "Create 2D Projection Grid."
            
        if self.current_inliers is None:
            self.status.text = "No highlighted plane to remove."
            return
        try:
            self.scene.scene.remove_geometry(self._path_geom_name)
        except Exception:
            pass

        if self.pcd_flag and self.pcd is not None:
            self.pcd = remove_planes_points(self.pcd, self.current_inliers)

            colors = np.tile(PCD_BASE_COLOR, (len(self.pcd.points), 1))
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            self.found_planes = []
            self.current_inliers = None

            self.status.text = "Removed plane from PCD. Click 'Select plane' again."
            self._replace_pcd()
            
            self._reset_after_pcd_change()

            return
    
        self.mesh = remove_planes_mesh(self.mesh, self.current_inliers)

        colors = np.tile(MESH_BASE_COLOR, (len(self.mesh.vertices), 1))
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.mesh.compute_vertex_normals()

        self.line_set = create_line_set_from_mesh(self.mesh, [0, 0, 0])

        self.found_planes = []
        self.current_inliers = None

        self.status.text = "Plane removed. Click 'Select plane' to compute again."
        self._replace_mesh_and_wire()
                
    def on_recompute_planes(self):
        """
        Re-run plane detection (RANSAC).
        """
            
        self.current_inliers = None
        if self.pcd_flag and self.pcd is not None:
            planes = plane_ransac_points(self.pcd)
            self.found_planes = [(p[1], p[3]) for p in planes]
            self.status.text = f"Recomputed {len(self.found_planes)} planes (PCD)."
            self._replace_pcd()
        else:
            planes = plane_ransac_mesh(self.mesh)
            self.found_planes = [(p[1], p[3]) for p in planes]
            self.status.text = f"Recomputed {len(self.found_planes)} planes (mesh)."
            self._replace_mesh_and_wire()

    # --- Clustering and door detection 
    def on_cluster(self):
        """
        Run DBSCAN on the sampled point cloud and color clusters for visualization.
        """
        if not self.pcd_flag or self.pcd is None:
            self.status.text = "Clustering requires point cloud (sample mesh first)."
            return

        pts = np.asarray(self.pcd.points)
        if pts.shape[0] == 0:
            self.status.text = "PCD is empty."
            return

        if self.path is not None:
            self._update_grid_image()
            self.scene.scene.remove_geometry(self._path_geom_name)
            
        labels = dbscan(pts, eps=CLUSTERING_EPS, min_samples=CLUSTERING_MIN_SAMPLES)
        labels = np.asarray(labels, dtype=np.int32)
        self.labels = labels

        unique = np.unique(labels)
        
        def color_for_k(k):
            h = (k * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
            return np.array([r, g, b], dtype=np.float64)

        point_colors = np.zeros((len(labels), 3), dtype=np.float64)
        point_colors[labels == -1] = np.array([0.0, 0.0, 0.0])

        cluster_ids = [u for u in unique if u != -1]
        self.cluster_ids = cluster_ids
        for i, cid in enumerate(cluster_ids):
            point_colors[labels == cid] = color_for_k(i)

        self.pcd.colors = o3d.utility.Vector3dVector(point_colors)

        self.status.text = f"DBSCAN done: {len(cluster_ids)} clusters. Black is noise."
        self._replace_pcd()
        
        self._reset_after_labels_change()

    def on_find_door_candidates(self):
        """
        Detect door-like clusters, mark candidate points in red and optionally remove candidate points above a height threshold. 
        """
        if self.path is not None:
            self._update_grid_image()
            self.scene.scene.remove_geometry(self._path_geom_name)
                
        if not self.pcd_flag or self.pcd is None:
            self.status.text = "Door detection needs point cloud mode."
            return
        if self.labels is None:
            self.status.text = "Run clustering first (DBSCAN)."
            return

        pts = np.asarray(self.pcd.points)
        labels = np.asarray(self.labels)

        if len(pts) != len(labels):
            self.status.text = "labels/points size mismatch (recluster)."
            return

        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        if Z_THRESHOLD is None:
            z_threshold_local = z_min + (z_max - z_min) * 2.0 / 3.0
        else:
            z_threshold_local = Z_THRESHOLD

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]

        colors = np.tile(PCD_BASE_COLOR, (len(pts), 1))

        candidate_infos = [] 

        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            cluster_pts = pts[idx]

            x_extent = np.ptp(cluster_pts[:, 0])
            z_extent = np.ptp(cluster_pts[:, 2])

            if not (MIN_DOOR_WIDTH <= x_extent <= MAX_DOOR_WIDTH and z_extent >= MIN_DOOR_HEIGHT):
                continue

            valid = idx[pts[idx, 2] <= z_threshold_local]
            if len(valid) < 3:
                continue

            score = float(x_extent)
            candidate_infos.append((score, int(lab), valid, idx))

            colors[valid] = DOOR_CANDIDATE_COLOR

        if not candidate_infos:
            self.status.text = "No door candidates found."
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            self._replace_pcd()
            return

        candidate_infos.sort(reverse=True, key=lambda t: t[0])
        _, best_lab, best_valid, _ = candidate_infos[0]

        best_xy, hull = None, None
        try:            
            best_xy = pts[best_valid][:, :2]
            hull = ConvexHull(best_xy)
        except Exception as e:
            self.status.text = f"Door candidate found, but hull failed: {e}"

        remove_mask = np.zeros(len(pts), dtype=bool)
        for _, lab, _, idx in candidate_infos:
            remove_mask[idx] |= (pts[idx, 2] > z_threshold_local)

        keep_mask = ~remove_mask
        pts2 = pts[keep_mask]
        labels2 = labels[keep_mask]
        colors2 = colors[keep_mask]

        self.pcd.points = o3d.utility.Vector3dVector(pts2)
        self.pcd.colors = o3d.utility.Vector3dVector(colors2)
                
        self.status.text = f"Door candidates: {len(candidate_infos)} (best label={int(best_lab)})."
        
        self.found_planes = []
        self.current_inliers = None
        self._replace_pcd()    
        
        self._reset_after_pcd_change() 
        
        self.labels = labels2  
        self.door_label = int(best_lab)
        if best_xy is not None and hull is not None:
            self.door_hull_points = best_xy[hull.vertices]
        else:
            self.door_hull_points = None
    
    # --- Grid creation, rendering and interaction ---
    def _ignore_mouse(self, event):
        """ Ignore all mouse events on the grid ImageWidget when grid picking is disabled."""
        return gui.Widget.EventCallbackResult.IGNORED

    def on_grid_toggle(self):
        """
        Toggle grid interaction.
        """
        grid_flag = self.on_create_grid() 
        if not grid_flag:
            return
        self.grid_active = True
        self.grid_img.set_on_mouse(self.on_grid_mouse)
        self.btn_grid.text = "Reset grid (re-pick start/goal)."
        self.status.text = "Grid created. Click start then goal."
        
        if self.path is not None:
            self._update_grid_image()
            self.scene.scene.remove_geometry(self._path_geom_name)
        
    def on_create_grid(self) -> bool:
        """
        Build the 2D occupancy grid from the current point cloud and clusters,
        expand obstacles using the current radius, clear any previous selected points 
        and render the result in the grid ImageWidget.
        
        Returns:
            True if the grid was created/ updated, False if PCD/clustering is missing.
        """
        if not self.pcd_flag or self.pcd is None:
            self.status.text = "Grid needs point cloud mode. Sample mesh first."
            return False
        if self.labels is None:
            self.status.text = "Run clustering first."
            return False
        
        if self.occupancy_grid is None:
            occ, x_min, y_min, x_max, y_max = create_occupancy_grid(
                self.pcd, self.labels, resolution=self.grid_resolution, z_threshold=Z_THRESHOLD
            )
            self.occupancy_grid = occ
            self.x_min, self.y_min = x_min, y_min
            self.x_max, self.y_max = x_max, y_max

        self.expanded_grid = expand_obstacles(
            self.occupancy_grid,
            radius=self.radius,
            resolution=self.grid_resolution
        )
        self._reset_after_grid_change()
        self._update_grid_image()        
        
        return True

    def _update_grid_image(self):
        """
        Render the expanded occupancy grid as an RGB image and display it in the ImageWidget.
        """
        if self.expanded_grid is None:
            return

        gt = np.fliplr(self.expanded_grid).T   
        h, w = gt.shape

        img = np.empty((h, w, 3), dtype=np.uint8)
        img[gt == 0] = (0, 0, 0)
        img[gt == 1] = (255, 255, 255)
        img[gt == 2] = (160, 160, 160)
        img[gt == 3] = (255, 0, 0)

        img = np.ascontiguousarray(img)

        step = max(1, int(np.ceil(max(h, w) / MAX_TEX)))
        self._grid_step = step
        img = np.ascontiguousarray(img[::step, ::step, :])

        self._grid_base_img = img.copy()
        self.grid_img.update_image(o3d.geometry.Image(img))
    
    def on_radius_changed(self, value):
        """
        Update expansion radius and redraw the expanded grid.
        """
        self.radius = float(value)
        self.radius_label.text = f"Radius: {self.radius:.2f} m"

        if self.occupancy_grid is None:
            return

        if self.path is not None:
            self._update_grid_image()
            self.scene.scene.remove_geometry(self._path_geom_name)
                
        self.expanded_grid = expand_obstacles(
            self.occupancy_grid,
            radius=self.radius,
            resolution=self.grid_resolution
        )
        self._update_grid_image()
        self._reset_after_grid_change()
        
    def on_grid_mouse(self, event):
        """
        Handle mouse clicks on the 2D occupancy grid ImageWidget and map them to start/goal points.
        """
        if event.type != gui.MouseEvent.Type.BUTTON_DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.buttons == 0:
            return gui.Widget.EventCallbackResult.IGNORED
        if not self.grid_active or self.expanded_grid is None:
            return gui.Widget.EventCallbackResult.IGNORED

        frame = self.grid_img.frame
        if frame.width <= 1 or frame.height <= 1:
            return gui.Widget.EventCallbackResult.IGNORED

        x = int(event.x - frame.x)
        y = int(event.y - frame.y)
        if x < 0 or y < 0 or x >= frame.width or y >= frame.height:
            return gui.Widget.EventCallbackResult.IGNORED

        gt_h, gt_w, _ = self._grid_base_img.shape 
        ix = int(x * gt_w / frame.width)
        iy = int((frame.height - 1 - y) * gt_h / frame.height)  

        ix = max(0, min(gt_w - 1, ix))
        iy = max(0, min(gt_h - 1, iy))

        x_grid = ix
        y_grid = self.expanded_grid.shape[1] - 1 - iy

        if np.fliplr(self.expanded_grid)[x_grid, y_grid] != 1:
            self.status.text = "Pick a free (white) cell."
            return gui.Widget.EventCallbackResult.HANDLED

        if self.start_point_grid is None:
            self.start_point_grid = (x_grid, y_grid)
            self.start_point_world = np.array([
                self.x_min + x_grid * self.grid_resolution,
                self.y_min + y_grid * self.grid_resolution
            ])
            self.status.text = f"Start set: {self.start_point_grid}"
        elif self.goal_point_grid is None:
            self.goal_point_grid = (x_grid, y_grid)
            self.goal_point_world = np.array([
                self.x_min + x_grid * self.grid_resolution,
                self.y_min + y_grid * self.grid_resolution
            ])
            self.status.text = f"Goal set: {self.goal_point_grid}"
        self._redraw_grid_with_points()
        return gui.Widget.EventCallbackResult.HANDLED
    
    def _draw_disc(self, img, cx, cy, color, r=MARKER_SIZE):
        """
        Draw a filled circular marker onto the grid image. 
        """
        h, w = img.shape[:2]
        x0, x1 = max(0, cx - r), min(w - 1, cx + r)
        y0, y1 = max(0, cy - r), min(h - 1, cy + r)
        yy, xx = np.ogrid[y0:y1+1, x0:x1+1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[y0:y1+1, x0:x1+1][mask] = color

    def _redraw_grid_with_points(self):
        """
        Redraw the grid image and overlay the currently selected start/goal markers.
        """
        if self._grid_base_img is None:
            return
        img = self._grid_base_img.copy()

        if self.start_point_grid is not None:
            sx, sy = self.start_point_grid
            self._draw_disc(img, sx, sy, (0, 0, 255), r=8)  

        if self.goal_point_grid is not None:
            gx, gy = self.goal_point_grid
            self._draw_disc(img, gx, gy, (255, 0, 0), r=8) 

        self.grid_img.update_image(o3d.geometry.Image(np.ascontiguousarray(img)))
    
    # --- Path planning (A*)
    def on_find_path(self):
        """
        Compute an A* path on the current expanded occupancy grid and visualize it:
        If goal is not selected, try to auto-pick a goal near the detected door.
        Draw the path on the 2D grid widget and as a 3D overlay in the scene.
        """

        if self.expanded_grid is None:
            self.status.text = "Create grid first."
            return
        if self.start_point_grid is None:
            self.status.text = "Pick a start point on the grid."
            return

        if self.goal_point_grid is None:
            if self.door_hull_points is None:
                self.status.text = "Pick a goal or run door detection first."
                return

            goal = self._pick_goal_near_door()
            if goal is None:
                self.status.text = "Could not find a free goal near door."
                return
            self.goal_point_grid = goal
            self.status.text = f"Auto goal set: {self.goal_point_grid}"

        path = a_star_search(np.fliplr(self.expanded_grid), self.start_point_grid, self.goal_point_grid)
        self.path = path
        
        if path is None or len(path) == 0:
            self.status.text = "No path found."
            return

        self._draw_path_on_grid_widget(path)

        self._draw_path_in_3d(path)

        self.status.text = f"Path found! length={len(path)}"

    def _pick_goal_near_door(self):
        """
        Pick a free goal near the detected door hull.
        """
        if self.expanded_grid is None or self.door_hull_points is None:
            return None
        
        grid_plan = np.fliplr(self.expanded_grid)
        
        def is_available(point):
            x, y = point
            if 0 <= x < grid_plan.shape[0] and 0 <= y < grid_plan.shape[1]:
                return grid_plan[x, y] == 1
            return False

        directions = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        expand_cells = int(np.ceil(self.radius / self.grid_resolution)) + 1

        door = self.door_hull_points.copy()
        door = door[np.argsort(door[:, 1])] 

        for point in door:
            i = int((point[0] - self.x_min) / self.grid_resolution)
            j = int((point[1] - self.y_min) / self.grid_resolution)

            for dx, dy in directions:
                new_point = (i + dx * expand_cells, j + dy * expand_cells)
                if is_available(new_point):
                    return new_point

        return None

    def _draw_path_on_grid_widget(self, path, thickness=PATH_THICKNESS):
        """
        Overlay the computed path and start/goal markers on the displayed occupancy grid.
        """
        if self._grid_base_img is None:
            self._update_grid_image()
            if self._grid_base_img is None:
                return
        
        img = self._grid_base_img.copy()
        step = max(1, int(self._grid_step))
        w, h = img.shape[:2]

        def paint_point(xg, yg, color):
            col = int(xg // step)   
            row = int(yg // step)

            y0 = max(0, row - thickness)
            y1 = min(h, row + thickness + 1)
            x0 = max(0, col - thickness)
            x1 = min(w, col + thickness + 1)
            
            img[x0:x1, y0:y1] = color
        
        for (yg, xg) in path:
            paint_point(xg, yg, PATH_COLOR) 

        if self.start_point_grid is not None:
            paint_point(*self.start_point_grid[::-1], START_COLOR)  
        if self.goal_point_grid is not None:
            paint_point(*self.goal_point_grid[::-1], GOAL_COLOR) 
        
        self.grid_img.update_image(o3d.geometry.Image(np.ascontiguousarray(img)))    

  
    def _draw_path_in_3d(self, path):
        """
        Render the computed path as a green point-cloud overlay in the 3D scene.
        """
        if self.x_min is None or self.y_max is None or not path:
            return
        
        try:
            self.scene.scene.remove_geometry(self._path_geom_name)
        except Exception:
            pass
                
        z_min = 0.0
        if self.pcd_flag and self.pcd is not None:
            pts = np.asarray(self.pcd.points)
            if pts.size > 0:
                z_min = float(pts[:, 2].min())

        world_path = []
        for (xg, yg) in path:
            xw = self.x_min + xg * self.grid_resolution
            yw = self.y_max - yg * self.grid_resolution
            world_path.append([xw, yw, z_min])

        world_path = np.asarray(world_path, dtype=np.float64)

        path_pcd = o3d.geometry.PointCloud()
        path_pcd.points = o3d.utility.Vector3dVector(world_path)
        path_pcd.paint_uniform_color([0.0, 1.0, 0.0])

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = PATH_POINT_SIZE

        self.scene.scene.add_geometry(self._path_geom_name, path_pcd, mat)

def main(mesh_path):    

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    colors = np.tile([0.75, 0.75, 0.75], (len(mesh.vertices), 1))

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    line_set = create_line_set_from_mesh(mesh, [0, 0, 0])  

    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height

    gui.Application.instance.initialize()
    _app = MeshApp(mesh, line_set, screen_width, screen_height)
    gui.Application.instance.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_path", nargs="?", default="RoomMesh.ply")
    args = parser.parse_args()
    main(args.mesh_path)