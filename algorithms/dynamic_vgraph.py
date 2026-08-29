
from scenario.geometry import navigation_space


class DynamicVGraph:
    def __init__(self, height, width, occupancy, agv_radius):
        
        self.height = height
        self.width = width
        self.occupancy = occupancy
        self.agv_radius = agv_radius
        self.graph = None
        
        # mappings between grid cells and graph edges
        self.cell_to_edge = None
        self.edge_to_cell = None
        
    
        
        
    def preprocessing(self):

        
        free_space, obstacle_space = navigation_space(self.occupancy, self.agv_radius, buffer_segments=0)


        

        
        # extract convex vertices from each shape
        
        # for each vertex pair determine which cells the edge between them crosses
        
        # create graph with vertices and edges, store mappings between cells and edges
        
        
        