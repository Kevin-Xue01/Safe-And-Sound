import json


class Database:
    def __init__(self):
        with open("config.json", "r") as file:
            self.data = json.load(file, parse_int=None)

    def get_warp_perspective_data(self):
        return self.data["warp_perspective"]

    def get_green_mask_data(self):
        return self.data["green_mask"]

    def get_green_polygon_data(self):
        return self.data["green_polygon"]

    def get_ball_edge_detection_data(self):
        return self.data["ball_edge_detection"]

    def get_ball_polygon_data(self):
        return self.data["ball_polygon"]
