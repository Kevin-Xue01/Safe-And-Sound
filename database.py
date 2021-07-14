import json


class Database:
    def __init__(self):
        with open("config.json", "r") as file:
            self.data = json.load(file, parse_int=None)

    def get_warp_perspective_data(self):
        return self.data["warp_perspective"]

    def update_warp_perspective_data(self, data):
        with open("config.json", "w") as file:
            self.data["warp_perspective"] = data
            json.dump(self.data, file, indent=4)

    def get_rectangle_mask_data(self):
        return self.data["rectangle_mask"]

    def update_rectangle_mask_data(self, data):
        with open("config.json", "w") as file:
            self.data["rectangle_mask"] = data
            json.dump(self.data, file, indent=4)

    def get_rectangle_polygon_contour_data(self):
        return self.data["rectangle_polygon_contour"]

    def update_rectangle_polygon_contour_data(self, data):
        with open("config.json", "w") as file:
            self.data["rectangle_polygon_contour"] = data
            json.dump(self.data, file, indent=4)

    def get_ball_edge_detection_data(self):
        return self.data["ball_edge_detection"]

    def update_ball_edge_detection_data(self, data):
        with open("config.json", "w") as file:
            self.data["ball_edge_detection"] = data
            json.dump(self.data, file, indent=4)

    def get_ball_polygon_data(self):
        return self.data["ball_polygon"]

    def update_ball_polygon_data(self, data):
        with open("config.json", "w") as file:
            self.data["ball_polygon"] = data
            json.dump(self.data, file, indent=4)

    def get_ball_mask_data(self):
        return self.data["ball_mask"]

    def update_ball_mask_data(self, data):
        with open("config.json", "w") as file:
            self.data["ball_mask"] = data
            json.dump(self.data, file, indent=4)
